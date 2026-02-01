"""
Audio-conditioned image-to-video pipeline for lip-sync generation.

This pipeline extends TI2VidTwoStagesPipeline with audio input conditioning,
allowing video generation that is synchronized to provided audio (lip-sync).
The input audio is encoded to latents and used as the initial audio state,
with the denoise mask set to preserve the audio while generating synchronized video.
"""

import logging
from collections.abc import Iterator
from dataclasses import replace

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps, SingleGPUModelBuilder
from ltx_core.model.audio_vae import AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER, AudioEncoderConfigurator
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.audio_vae.ops import AudioProcessor
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.text_encoders.gemma import encode_text
from ltx_core.tools import AudioLatentTools
from ltx_core.types import AudioLatentShape, LatentState, VideoPixelShape
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.args import default_2_stage_arg_parser
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE, STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    euler_denoising_loop,
    generate_enhanced_prompt,
    get_device,
    image_conditionings_by_replacing_latent,
    multi_modal_guider_denoising_func,
    noise_video_state,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()


class AudioConditionedI2VPipeline:
    """
    Audio-conditioned image-to-video pipeline for lip-sync generation.

    This pipeline generates video synchronized to input audio. The audio is encoded
    to latents and kept fixed during denoising, while the video generation attends
    to the audio through cross-modal attention for lip-sync.

    Unlike text-to-audio-video pipelines, this pipeline:
    1. Takes audio input (waveform or file) instead of generating audio
    2. Encodes audio to latents with low noise (preserving original audio)
    3. Generates video that synchronizes with the fixed audio
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: str = device,
        fp8transformer: bool = False,
    ):
        self.device = device
        self.dtype = torch.bfloat16

        self.stage_1_model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=loras,
            fp8transformer=fp8transformer,
        )

        self.stage_2_model_ledger = self.stage_1_model_ledger.with_loras(loras=distilled_lora)

        self.pipeline_components = PipelineComponents(dtype=self.dtype, device=device)

        self.checkpoint_path = checkpoint_path
        self._audio_encoder = None
        self._audio_processor = None

    def _get_audio_encoder(self) -> tuple:
        """Lazily load audio encoder and processor."""
        if self._audio_encoder is None:
            self._audio_encoder = SingleGPUModelBuilder(
                model_path=self.checkpoint_path,
                model_class_configurator=AudioEncoderConfigurator,
                model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            ).build(device=torch.device(self.device), dtype=torch.float32)

            self._audio_processor = AudioProcessor(
                sample_rate=self._audio_encoder.sample_rate,
                mel_bins=self._audio_encoder.mel_bins,
                mel_hop_length=self._audio_encoder.mel_hop_length,
                n_fft=self._audio_encoder.n_fft,
            ).to(self.device)

        return self._audio_encoder, self._audio_processor

    def encode_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        target_duration: float,
    ) -> torch.Tensor:
        """
        Encode audio waveform to latent representation.

        Args:
            waveform: Audio waveform tensor of shape [channels, samples] or [batch, channels, samples]
            sample_rate: Sample rate of the input waveform
            target_duration: Target duration in seconds (will pad/trim audio to match)

        Returns:
            Audio latents tensor of shape [batch, channels, time, freq]
        """
        audio_encoder, audio_processor = self._get_audio_encoder()

        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        target_samples = int(target_duration * sample_rate)
        current_samples = waveform.shape[-1]

        if current_samples > target_samples:
            waveform = waveform[..., :target_samples]
        elif current_samples < target_samples:
            pad_size = target_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))

        if waveform.shape[1] == 1:
            waveform = waveform.repeat(1, 2, 1)

        waveform = waveform.to(device=self.device, dtype=torch.float32)
        mel = audio_processor.waveform_to_mel(waveform, waveform_sample_rate=sample_rate)

        with torch.no_grad():
            latents = audio_encoder(mel)

        return latents.to(self.dtype)

    def _create_audio_conditioned_state(
        self,
        audio_latent: torch.Tensor,
        output_shape: VideoPixelShape,
        noise_strength: float = 0.1,
        generator: torch.Generator | None = None,
    ) -> tuple[LatentState, AudioLatentTools]:
        """
        Create audio state with conditioning from input audio.

        Args:
            audio_latent: Pre-encoded audio latents [batch, channels, time, freq]
            output_shape: Target video shape (used to compute audio latent shape)
            noise_strength: Amount of noise to add (0.0 = keep audio exact, 1.0 = full noise)
            generator: Random generator for reproducibility

        Returns:
            Tuple of (LatentState, AudioLatentTools) with audio conditioned state
        """
        audio_latent_shape = AudioLatentShape.from_video_pixel_shape(output_shape)
        audio_tools = AudioLatentTools(self.pipeline_components.audio_patchifier, audio_latent_shape)

        target_shape = audio_latent_shape.to_torch_shape()
        if audio_latent.shape != target_shape:
            b, c, t, f = audio_latent.shape
            target_b, target_c, target_t, target_f = target_shape

            if t > target_t:
                audio_latent = audio_latent[:, :, :target_t, :]
            elif t < target_t:
                audio_latent = torch.nn.functional.pad(audio_latent, (0, 0, 0, target_t - t))

            if f != target_f or c != target_c:
                logging.warning(f"Audio latent shape mismatch: {audio_latent.shape} vs target {target_shape}")

        state = audio_tools.create_initial_state(
            device=self.device,
            dtype=self.dtype,
            initial_latent=audio_latent,
        )

        denoise_mask = torch.full_like(state.denoise_mask, noise_strength)
        state = replace(state, denoise_mask=denoise_mask, clean_latent=state.latent.clone())

        if noise_strength > 0 and generator is not None:
            noise = torch.randn_like(state.latent, generator=generator)
            state = replace(state, latent=state.latent + noise * noise_strength)

        return state, audio_tools

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str,
        audio_waveform: torch.Tensor,
        audio_sample_rate: int,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams,
        audio_guider_params: MultiModalGuiderParams,
        images: list[tuple[str, int, float]],
        audio_noise_strength: float = 0.1,
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        """
        Generate video synchronized to input audio.

        Args:
            prompt: Text prompt describing the video content
            negative_prompt: Negative prompt for guidance
            audio_waveform: Input audio tensor [channels, samples]
            audio_sample_rate: Sample rate of input audio
            seed: Random seed for reproducibility
            height: Output video height (must be divisible by 64)
            width: Output video width (must be divisible by 64)
            num_frames: Number of frames to generate
            frame_rate: Output frame rate
            num_inference_steps: Number of diffusion steps (stage 1)
            video_guider_params: Guidance parameters for video
            audio_guider_params: Guidance parameters for audio (lower values recommended)
            images: List of (image_path, frame_idx, strength) for image conditioning
            audio_noise_strength: Noise added to audio latent (0.0-1.0, lower = more faithful)
            tiling_config: Optional tiling config for decoding
            enhance_prompt: Whether to enhance prompt with Gemma

        Returns:
            Tuple of (video_iterator, audio_tensor)
        """
        assert_resolution(height=height, width=width, is_two_stage=True)

        duration_s = num_frames / frame_rate

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16

        audio_latent = self.encode_audio(
            waveform=audio_waveform,
            sample_rate=audio_sample_rate,
            target_duration=duration_s,
        )

        text_encoder = self.stage_1_model_ledger.text_encoder()
        if enhance_prompt:
            prompt = generate_enhanced_prompt(
                text_encoder, prompt, images[0][0] if len(images) > 0 else None, seed=seed
            )
        context_p, context_n = encode_text(text_encoder, prompts=[prompt, negative_prompt])
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n

        torch.cuda.synchronize()
        del text_encoder
        cleanup_memory()

        video_encoder = self.stage_1_model_ledger.video_encoder()
        transformer = self.stage_1_model_ledger.transformer()
        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)

        def first_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=multi_modal_guider_denoising_func(
                    video_guider=MultiModalGuider(params=video_guider_params, negative_context=v_context_n),
                    audio_guider=MultiModalGuider(params=audio_guider_params, negative_context=a_context_n),
                    v_context=v_context_p,
                    a_context=a_context_p,
                    transformer=transformer,
                ),
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=frame_rate
        )
        stage_1_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )

        video_state, video_tools = noise_video_state(
            output_shape=stage_1_output_shape,
            noiser=noiser,
            conditionings=stage_1_conditionings,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
        )

        audio_state, audio_tools = self._create_audio_conditioned_state(
            audio_latent=audio_latent,
            output_shape=stage_1_output_shape,
            noise_strength=audio_noise_strength,
            generator=generator,
        )

        video_state, audio_state = first_stage_denoising_loop(sigmas, video_state, audio_state, stepper)

        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)

        torch.cuda.synchronize()
        del transformer
        cleanup_memory()

        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=self.stage_2_model_ledger.spatial_upsampler(),
        )

        torch.cuda.synchronize()
        cleanup_memory()

        transformer = self.stage_2_model_ledger.transformer()
        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)

        def second_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=v_context_p, audio_context=a_context_p, transformer=transformer
                ),
            )

        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )

        video_state, video_tools = noise_video_state(
            output_shape=stage_2_output_shape,
            noiser=noiser,
            conditionings=stage_2_conditionings,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=distilled_sigmas[0],
            initial_latent=upscaled_video_latent,
        )

        audio_state, audio_tools = self._create_audio_conditioned_state(
            audio_latent=audio_state.latent,
            output_shape=stage_2_output_shape,
            noise_strength=audio_noise_strength * 0.5,
            generator=generator,
        )

        video_state, audio_state = second_stage_denoising_loop(distilled_sigmas, video_state, audio_state, stepper)

        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)

        torch.cuda.synchronize()
        del transformer, video_encoder
        cleanup_memory()

        # Clone latents to escape inference mode (required for VAE decoder compatibility)
        video_latent = video_state.latent.clone()
        audio_latent = audio_state.latent.clone()

        decoded_video = vae_decode_video(
            video_latent, self.stage_2_model_ledger.video_decoder(), tiling_config, generator
        )
        decoded_audio = vae_decode_audio(
            audio_latent, self.stage_2_model_ledger.audio_decoder(), self.stage_2_model_ledger.vocoder()
        )

        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    import argparse

    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Audio-conditioned video generation")
    parser.add_argument("--checkpoint-path", required=True, help="Path to LTX-2 checkpoint")
    parser.add_argument("--distilled-lora", required=True, help="Path to distilled LoRA")
    parser.add_argument("--spatial-upsampler-path", required=True, help="Path to spatial upsampler")
    parser.add_argument("--gemma-root", required=True, help="Path to Gemma model")
    parser.add_argument("--audio-path", required=True, help="Path to input audio file")
    parser.add_argument("--image-path", required=True, help="Path to conditioning image")
    parser.add_argument("--output-path", required=True, help="Path to output video")
    parser.add_argument("--prompt", default="A person speaking naturally", help="Text prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--height", type=int, default=896, help="Output height")
    parser.add_argument("--width", type=int, default=512, help="Output width")
    parser.add_argument("--num-frames", type=int, default=121, help="Number of frames")
    parser.add_argument("--frame-rate", type=float, default=24.0, help="Frame rate")
    parser.add_argument("--audio-noise-strength", type=float, default=0.1, help="Audio noise (0.0-1.0)")

    args = parser.parse_args()

    import torchaudio

    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.components.guiders import MultiModalGuiderParams

    waveform, sample_rate = torchaudio.load(args.audio_path)

    distilled_lora = [LoraPathStrengthAndSDOps(args.distilled_lora, 0.6, LTXV_LORA_COMFY_RENAMING_MAP)]

    pipeline = AudioConditionedI2VPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=[],
        fp8transformer=True,
    )

    video_guider = MultiModalGuiderParams(
        cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7, modality_scale=3.0, stg_blocks=[29]
    )
    audio_guider = MultiModalGuiderParams(
        cfg_scale=2.0, stg_scale=0.5, rescale_scale=0.5, modality_scale=2.0, stg_blocks=[29]
    )

    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt="worst quality, blurry, jittery, distorted, mismatched lip sync",
        audio_waveform=waveform,
        audio_sample_rate=sample_rate,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=40,
        video_guider_params=video_guider,
        audio_guider_params=audio_guider,
        images=[(args.image_path, 0, 1.0)],
        audio_noise_strength=args.audio_noise_strength,
    )

    tiling_config = TilingConfig.default()
    video_chunks = get_video_chunks_number(args.num_frames, tiling_config)

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=args.output_path,
        video_chunks_number=video_chunks,
    )

    logging.info(f"Video saved to {args.output_path}")


if __name__ == "__main__":
    main()
