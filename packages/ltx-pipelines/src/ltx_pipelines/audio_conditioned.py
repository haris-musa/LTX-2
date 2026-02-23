"""Audio-conditioned video generation pipeline with optional upscaling."""

import logging
from collections.abc import Iterator
from dataclasses import replace

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps, SingleGPUModelBuilder
from ltx_core.model.audio_vae import AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER, AudioEncoderConfigurator
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.audio_vae.ops import AudioProcessor
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.quantization import QuantizationPolicy
from ltx_core.text_encoders.gemma import encode_text
from ltx_core.tools import AudioLatentTools
from ltx_core.types import AudioLatentShape, LatentState, VideoPixelShape
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    euler_denoising_loop,
    get_device,
    image_conditionings_by_replacing_latent,
    multi_modal_guider_denoising_func,
    noise_video_state,
    simple_denoising_func,
)
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()
logger = logging.getLogger(__name__)


def _log_vram(message: str) -> None:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(message, torch.cuda.memory_allocated() / 1e9)


class AudioConditionedI2VPipeline:
    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps] | None = None,
        distilled_lora: list[LoraPathStrengthAndSDOps] | None = None,
        spatial_upsampler_path: str | None = None,
        device: str = device,
        quantization: QuantizationPolicy | None = None,
    ):
        self.device = device
        self.dtype = torch.bfloat16
        self.checkpoint_path = checkpoint_path
        self.spatial_upsampler_path = spatial_upsampler_path

        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=loras or [],
            quantization=quantization,
        )

        self.stage_2_model_ledger = None
        if distilled_lora and spatial_upsampler_path:
            self.stage_2_model_ledger = self.model_ledger.with_loras(loras=distilled_lora)

        self.pipeline_components = PipelineComponents(dtype=self.dtype, device=device)
        self._audio_encoder = None
        self._audio_processor = None
        self._text_encoder = None
        self._video_encoder = None
        self._stage_1_transformer = None
        self._stage_2_transformer = None
        self._stage_2_upsampler = None
        self._stage_1_video_decoder = None
        self._stage_2_video_decoder = None
        self._stage_1_audio_decoder = None
        self._stage_2_audio_decoder = None
        self._stage_1_vocoder = None
        self._stage_2_vocoder = None

    def _get_audio_encoder(self) -> tuple:
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

    def _get_text_encoder(self):
        if self._text_encoder is None:
            self._text_encoder = self.model_ledger.text_encoder()
        return self._text_encoder

    def _get_video_encoder(self):
        if self._video_encoder is None:
            self._video_encoder = self.model_ledger.video_encoder()
        return self._video_encoder

    def _get_stage_1_transformer(self):
        if self._stage_1_transformer is None:
            self._stage_1_transformer = self.model_ledger.transformer()
        return self._stage_1_transformer

    def _get_stage_2_transformer(self):
        if self.stage_2_model_ledger is None:
            raise ValueError("Stage 2 transformer is unavailable without distilled_lora and spatial_upsampler_path")
        if self._stage_2_transformer is None:
            self._stage_2_transformer = self.stage_2_model_ledger.transformer()
        return self._stage_2_transformer

    def _get_stage_2_upsampler(self):
        if self.stage_2_model_ledger is None:
            raise ValueError("Stage 2 upsampler is unavailable without distilled_lora and spatial_upsampler_path")
        if self._stage_2_upsampler is None:
            self._stage_2_upsampler = self.stage_2_model_ledger.spatial_upsampler()
        return self._stage_2_upsampler

    def _get_video_decoder(self, use_upscaler: bool):
        if use_upscaler:
            if self.stage_2_model_ledger is None:
                raise ValueError("Stage 2 video decoder is unavailable without distilled_lora and spatial_upsampler_path")
            if self._stage_2_video_decoder is None:
                self._stage_2_video_decoder = self.stage_2_model_ledger.video_decoder()
            return self._stage_2_video_decoder
        if self._stage_1_video_decoder is None:
            self._stage_1_video_decoder = self.model_ledger.video_decoder()
        return self._stage_1_video_decoder

    def _get_audio_decoder(self, use_upscaler: bool):
        if use_upscaler:
            if self.stage_2_model_ledger is None:
                raise ValueError("Stage 2 audio decoder is unavailable without distilled_lora and spatial_upsampler_path")
            if self._stage_2_audio_decoder is None:
                self._stage_2_audio_decoder = self.stage_2_model_ledger.audio_decoder()
            return self._stage_2_audio_decoder
        if self._stage_1_audio_decoder is None:
            self._stage_1_audio_decoder = self.model_ledger.audio_decoder()
        return self._stage_1_audio_decoder

    def _get_vocoder(self, use_upscaler: bool):
        if use_upscaler:
            if self.stage_2_model_ledger is None:
                raise ValueError("Stage 2 vocoder is unavailable without distilled_lora and spatial_upsampler_path")
            if self._stage_2_vocoder is None:
                self._stage_2_vocoder = self.stage_2_model_ledger.vocoder()
            return self._stage_2_vocoder
        if self._stage_1_vocoder is None:
            self._stage_1_vocoder = self.model_ledger.vocoder()
        return self._stage_1_vocoder

    def encode_audio(self, waveform: torch.Tensor, sample_rate: int, target_duration: float) -> torch.Tensor:
        audio_encoder, audio_processor = self._get_audio_encoder()

        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        target_samples = int(target_duration * sample_rate)
        current_samples = waveform.shape[-1]

        if current_samples > target_samples:
            waveform = waveform[..., :target_samples]
        elif current_samples < target_samples:
            waveform = torch.nn.functional.pad(waveform, (0, target_samples - current_samples))

        if waveform.shape[1] == 1:
            waveform = waveform.repeat(1, 2, 1)

        waveform = waveform.to(device=self.device, dtype=torch.float32)
        mel = audio_processor.waveform_to_mel(waveform, waveform_sample_rate=sample_rate)

        with torch.no_grad():
            return audio_encoder(mel).to(self.dtype)

    def _create_audio_state(
        self,
        audio_latent: torch.Tensor,
        output_shape: VideoPixelShape,
        noise_strength: float,
        generator: torch.Generator | None,
    ) -> tuple[LatentState, AudioLatentTools]:
        audio_latent_shape = AudioLatentShape.from_video_pixel_shape(output_shape)
        audio_tools = AudioLatentTools(self.pipeline_components.audio_patchifier, audio_latent_shape)

        target_shape = audio_latent_shape.to_torch_shape()
        if audio_latent.shape != target_shape:
            _, _, t, _ = audio_latent.shape
            _, _, target_t, _ = target_shape
            if t > target_t:
                audio_latent = audio_latent[:, :, :target_t, :]
            elif t < target_t:
                audio_latent = torch.nn.functional.pad(audio_latent, (0, 0, 0, target_t - t))

        state = audio_tools.create_initial_state(device=self.device, dtype=self.dtype, initial_latent=audio_latent)
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
        use_upscaler: bool = False,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        if use_upscaler and self.stage_2_model_ledger is None:
            raise ValueError("use_upscaler=True requires distilled_lora and spatial_upsampler_path")

        assert_resolution(height=height, width=width, is_two_stage=use_upscaler)

        duration_s = num_frames / frame_rate
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        audio_latent = self.encode_audio(audio_waveform, audio_sample_rate, duration_s)

        text_encoder = self._get_text_encoder()
        context_p, context_n = encode_text(text_encoder, prompts=[prompt, negative_prompt])
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n

        _log_vram("VRAM before transformer build: %.2fGB")
        video_encoder = self._get_video_encoder()
        transformer = self._get_stage_1_transformer()
        _log_vram("VRAM after transformer build: %.2fGB")
        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)

        if use_upscaler:
            stage_1_shape = VideoPixelShape(batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=frame_rate)
        else:
            stage_1_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)

        conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_1_shape.height,
            width=stage_1_shape.width,
            video_encoder=video_encoder,
            dtype=self.dtype,
            device=self.device,
        )

        video_state, video_tools = noise_video_state(
            output_shape=stage_1_shape,
            noiser=noiser,
            conditionings=conditionings,
            components=self.pipeline_components,
            dtype=self.dtype,
            device=self.device,
        )

        audio_state, audio_tools = self._create_audio_state(
            audio_latent=audio_latent,
            output_shape=stage_1_shape,
            noise_strength=audio_noise_strength,
            generator=generator,
        )

        video_state, audio_state = euler_denoising_loop(
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
        _log_vram("VRAM after stage 1 denoising: %.2fGB")

        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)

        if use_upscaler:
            upsampler = self._get_stage_2_upsampler()
            upscaled_video = upsample_video(
                latent=video_state.latent[:1],
                video_encoder=video_encoder,
                upsampler=upsampler,
            )
            stage_1_audio_latent = audio_state.latent
            del video_state, audio_state

            transformer = self._get_stage_2_transformer()
            distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)

            stage_2_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
            stage_2_conds = image_conditionings_by_replacing_latent(
                images=images,
                height=height,
                width=width,
                video_encoder=video_encoder,
                dtype=self.dtype,
                device=self.device,
            )

            video_state, video_tools = noise_video_state(
                output_shape=stage_2_shape,
                noiser=noiser,
                conditionings=stage_2_conds,
                components=self.pipeline_components,
                dtype=self.dtype,
                device=self.device,
                noise_scale=distilled_sigmas[0],
                initial_latent=upscaled_video,
            )

            audio_state, audio_tools = self._create_audio_state(
                audio_latent=stage_1_audio_latent,
                output_shape=stage_2_shape,
                noise_strength=audio_noise_strength * 0.5,
                generator=generator,
            )

            video_state, audio_state = euler_denoising_loop(
                sigmas=distilled_sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=transformer,
                ),
            )

            video_state = video_tools.clear_conditioning(video_state)
            video_state = video_tools.unpatchify(video_state)
            audio_state = audio_tools.clear_conditioning(audio_state)
            audio_state = audio_tools.unpatchify(audio_state)

            video_latent = video_state.latent
            audio_latent_out = audio_state.latent
            del video_state, audio_state, stage_2_conds, upscaled_video, stage_1_audio_latent

        if not use_upscaler:
            video_latent = video_state.latent
            audio_latent_out = audio_state.latent

        logger.debug("Final Latent Shapes - Video: %s, Audio: %s", video_latent.shape, audio_latent_out.shape)
        logger.debug(
            "Audio Latent Stats - Max: %.4f, Min: %.4f, Mean: %.4f",
            audio_latent_out.max().item(),
            audio_latent_out.min().item(),
            audio_latent_out.mean().item(),
        )

        # Final preparation for decoding
        if not use_upscaler:
            del video_state, audio_state
        del v_context_p, a_context_p, v_context_n, a_context_n
        _log_vram("VRAM before decoder fetch: %.2fGB")

        video_decoder = self._get_video_decoder(use_upscaler=use_upscaler)
        _log_vram("VRAM after decoder build: %.2fGB")

        # Realize iterator immediately to prevent it from outliving local references
        decoded_chunks = list(vae_decode_video(video_latent, video_decoder, tiling_config, generator))
        decoded_video = torch.cat(decoded_chunks, dim=0) if len(decoded_chunks) > 1 else decoded_chunks[0]

        audio_decoder = self._get_audio_decoder(use_upscaler=use_upscaler)
        vocoder = self._get_vocoder(use_upscaler=use_upscaler)
        decoded_audio = vae_decode_audio(audio_latent_out, audio_decoder, vocoder)
        logger.debug(
            "Decoded Audio Stats - Max: %.4f, Min: %.4f",
            decoded_audio.max().item(),
            decoded_audio.min().item(),
        )

        return decoded_video, decoded_audio
