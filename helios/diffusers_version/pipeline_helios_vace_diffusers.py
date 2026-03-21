# Copyright 2025 The Helios Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HeliosVACEPipeline – VACE-conditioned autoregressive video generation.

Extends HeliosPipeline to support VACE (Video-to-Video with Attention Control
and Editing) conditioning.  The pipeline:

  1. Encodes a control video into per-chunk VACE latents (96ch):
       inactive (16ch) = VAE(video × (1 - mask))
       reactive (16ch) = VAE(video × mask)
       mask     (64ch) = spatially tiled downsampled mask
  2. At each AR chunk, slices the corresponding control latent window and
     passes it to the transformer via extra_transformer_kwargs.

Usage::

    from helios.diffusers_version.pipeline_helios_vace_diffusers import HeliosVACEPipeline

    pipe = HeliosVACEPipeline.from_pretrained(
        "BestWishYsh/Helios-Distilled",
        transformer=HeliosVACETransformer3DModel.from_pretrained(...)
    )
    pipe(...,
         control_video=video_tensor,      # [B, C, T, H, W] pixel values in [-1, 1]
         control_mask=mask_tensor,        # [B, 1, T, H, W] in [0, 1], 1 = edit region
         conditioning_scale=1.0,
    )
"""

from typing import Any, Callable

import torch
import torch.nn.functional as F

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import logging

from .pipeline_helios_diffusers import HeliosPipeline


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: torch.Generator | None = None,
    sample_mode: str = "sample",
) -> torch.Tensor:
    """Extract latents from VAE encoder output (not in diffusers.utils.torch_utils in this version)."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class HeliosVACEPipeline(HeliosPipeline):
    """HeliosPipeline extended with VACE conditioning.

    The transformer is expected to be a HeliosVACETransformer3DModel instance,
    but the pipeline also works with the plain HeliosTransformer3DModel (VACE
    kwargs are silently ignored if the transformer does not accept them).
    """

    # ---------------------------------------------------------------------- #
    # VACE latent preparation helpers
    # ---------------------------------------------------------------------- #

    def prepare_vace_latents(
        self,
        control_video: torch.Tensor,   # [B, C, T, H, W], pixel values
        control_mask: torch.Tensor,    # [B, 1, T, H, W], in [0, 1]
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Encode a control video into 96-channel VACE conditioning latents.

        Returns a tensor of shape [B, 96, T_lat, H_lat, W_lat].

        The 96 channels are:
          - ch  0..15 : inactive latents  = VAE(video * (1 - mask))
          - ch 16..31 : reactive latents  = VAE(video * mask)
          - ch 32..95 : 64-channel tiled mask (8×8 spatial tiles)
        """
        device = self._execution_device
        vae_dtype = self.vae.dtype

        latents_mean = torch.tensor(
            self.vae.config.latents_mean, device=device, dtype=torch.float32
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(
            self.vae.config.latents_std, device=device, dtype=torch.float32
        ).view(1, self.vae.config.z_dim, 1, 1, 1)

        video = control_video.to(device=device, dtype=vae_dtype)
        mask = control_mask.to(device=device, dtype=vae_dtype)
        mask = torch.where(mask > 0.5, torch.ones_like(mask), torch.zeros_like(mask))

        # Encode inactive (unmasked) and reactive (masked) regions separately
        inactive = video * (1 - mask)
        reactive = video * mask
        inactive_lat = retrieve_latents(self.vae.encode(inactive), generator, sample_mode="argmax")
        reactive_lat = retrieve_latents(self.vae.encode(reactive), generator, sample_mode="argmax")
        inactive_lat = ((inactive_lat.float() - latents_mean) * latents_std).to(vae_dtype)
        reactive_lat = ((reactive_lat.float() - latents_mean) * latents_std).to(vae_dtype)
        video_latents = torch.cat([inactive_lat, reactive_lat], dim=1)  # [B, 32, T_lat, H_lat, W_lat]

        # Build 64-channel mask at latent resolution (8×8 spatial tiling)
        B, _, T, H, W = mask.shape
        p_h = self.vae_scale_factor_spatial   # 8
        p_w = self.vae_scale_factor_spatial   # 8
        T_lat = (T + self.vae_scale_factor_temporal - 1) // self.vae_scale_factor_temporal
        H_lat = H // p_h
        W_lat = W // p_w

        # Rearrange mask to 64-channel representation: [p_h*p_w, T, H_lat, W_lat]
        mask_1c = mask[:, 0]  # [B, T, H, W]
        mask_tiled = mask_1c.view(B, T, H_lat, p_h, W_lat, p_w)
        mask_tiled = mask_tiled.permute(0, 3, 5, 1, 2, 4)          # [B, p_h, p_w, T, H_lat, W_lat]
        mask_tiled = mask_tiled.flatten(1, 2)                       # [B, 64, T, H_lat, W_lat]
        # Temporal downsampling to match latent frames
        mask_tiled = F.interpolate(
            mask_tiled.flatten(0, 1).unsqueeze(0),
            size=(T_lat, H_lat, W_lat),
            mode="nearest-exact",
        ).squeeze(0).view(B, 64, T_lat, H_lat, W_lat).to(vae_dtype)

        # Concatenate: [B, 96, T_lat, H_lat, W_lat]
        vace_latents = torch.cat([video_latents, mask_tiled], dim=1)
        return vace_latents

    # ---------------------------------------------------------------------- #
    # __call__ override
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def __call__(
        self,
        # ---- standard HeliosPipeline args ----
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] = None,
        height: int = 384,
        width: int = 640,
        num_frames: int = 132,
        num_inference_steps: int = 50,
        sigmas: list[float] = None,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        output_type: str | None = "np",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int], None] | PipelineCallback | MultiPipelineCallbacks | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
        # ---- I2V ----
        image: PipelineImageInput | None = None,
        image_latents: torch.Tensor | None = None,
        fake_image_latents: torch.Tensor | None = None,
        add_noise_to_image_latents: bool = True,
        image_noise_sigma_min: float = 0.111,
        image_noise_sigma_max: float = 0.135,
        # ---- V2V ----
        video: PipelineImageInput | None = None,
        video_latents: torch.Tensor | None = None,
        add_noise_to_video_latents: bool = True,
        video_noise_sigma_min: float = 0.111,
        video_noise_sigma_max: float = 0.135,
        # ---- Interactive ----
        use_interpolate_prompt: bool = False,
        interpolate_time_list: list = [7, 7, 7],
        interpolation_steps: int = 3,
        # ---- Stage 1 ----
        history_sizes: list = [16, 2, 1],
        num_latent_frames_per_chunk: int = 9,
        keep_first_frame: bool = True,
        is_skip_first_chunk: bool = False,
        # ---- Stage 2 ----
        is_enable_stage2: bool = False,
        pyramid_num_stages: int = 3,
        pyramid_num_inference_steps_list: list = [10, 10, 10],
        # ---- CFG Zero ----
        use_zero_init: bool | None = True,
        zero_steps: int | None = 1,
        # ---- DMD ----
        is_amplify_first_chunk: bool = False,
        # ---- VACE ----
        control_video: torch.Tensor | None = None,   # [B, C, T, H, W]
        control_mask: torch.Tensor | None = None,    # [B, 1, T, H, W]
        conditioning_scale: float = 1.0,
        # Feature 1: how to pad the last chunk when the control video doesn't fill
        #   the full window ("zero" = pad with zeros, "last_frame" = repeat last frame).
        vace_last_chunk_padding: str = "last_frame",
        # Feature 2: also inject VACE hints into the short-history token positions.
        inject_hints_to_history: bool = False,
        # Feature 3a: only inject hints at the pyramid stage whose resolution matches
        #   the native control-latent resolution; skip lower-res stages entirely.
        vace_only_inject_at_full_resolution: bool = False,
        # Feature 3b: run the VACE control stream at its native latent resolution and
        #   resize the output hints to each pyramid stage's token count.
        vace_process_at_full_resolution: bool = False,
    ):
        """Generate video with optional VACE conditioning.

        Args:
            control_video: Control video tensor, shape [B, C, T, H, W],
                pixel values expected in [-1, 1].  If None, VACE is disabled.
            control_mask: Binary mask tensor, shape [B, 1, T, H, W], values in
                [0, 1] where 1 marks the region to be edited/controlled.
                If None but control_video is provided, an all-ones mask is used
                (full-frame control, equivalent to style/motion transfer).
            conditioning_scale: Global scale factor for VACE hints (default 1.0).
        """
        # ------------------------------------------------------------------ #
        # If no control video, fall back to the unmodified base pipeline
        # ------------------------------------------------------------------ #
        if control_video is None:
            return super().__call__(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                num_videos_per_prompt=num_videos_per_prompt,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                output_type=output_type,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
                image=image,
                image_latents=image_latents,
                fake_image_latents=fake_image_latents,
                add_noise_to_image_latents=add_noise_to_image_latents,
                image_noise_sigma_min=image_noise_sigma_min,
                image_noise_sigma_max=image_noise_sigma_max,
                video=video,
                video_latents=video_latents,
                add_noise_to_video_latents=add_noise_to_video_latents,
                video_noise_sigma_min=video_noise_sigma_min,
                video_noise_sigma_max=video_noise_sigma_max,
                use_interpolate_prompt=use_interpolate_prompt,
                interpolate_time_list=interpolate_time_list,
                interpolation_steps=interpolation_steps,
                history_sizes=history_sizes,
                num_latent_frames_per_chunk=num_latent_frames_per_chunk,
                keep_first_frame=keep_first_frame,
                is_skip_first_chunk=is_skip_first_chunk,
                is_enable_stage2=is_enable_stage2,
                pyramid_num_stages=pyramid_num_stages,
                pyramid_num_inference_steps_list=pyramid_num_inference_steps_list,
                use_zero_init=use_zero_init,
                zero_steps=zero_steps,
                is_amplify_first_chunk=is_amplify_first_chunk,
            )

        # ------------------------------------------------------------------ #
        # Pre-encode the full control video into VACE latents
        # ------------------------------------------------------------------ #
        if control_mask is None:
            # All-ones mask: full-frame control
            B, C, T, H, W = control_video.shape
            control_mask = torch.ones(B, 1, T, H, W, device=control_video.device)

        vace_latents_full = self.prepare_vace_latents(
            control_video=control_video,
            control_mask=control_mask,
            generator=generator if not isinstance(generator, list) else generator[0],
        )
        # vace_latents_full: [B, 96, T_lat, H_lat, W_lat]

        # ------------------------------------------------------------------ #
        # Monkey-patch stage1_sample / stage2_sample so every call to the
        # transformer receives the right per-chunk VACE latent window.
        # ------------------------------------------------------------------ #
        # We compute the chunk window boundaries here because they mirror the
        # AR loop inside HeliosPipeline.__call__.
        vae_t = self.vae_scale_factor_temporal   # 4

        window_latent_frames = num_latent_frames_per_chunk
        T_lat_full = vace_latents_full.shape[2]

        # Store references so the closures below can access them
        _vace_latents = vace_latents_full
        _scale = conditioning_scale
        _chunk_counter = [0]   # mutable container for chunk index
        _num_history_frames = sum(history_sizes)

        # Save originals
        _orig_stage1 = self.stage1_sample.__func__
        _orig_stage2 = self.stage2_sample.__func__

        pipeline_self = self  # capture for closures

        def _build_vace_kwargs(chunk_idx: int) -> dict:
            """Build extra_transformer_kwargs for one AR chunk."""
            start = chunk_idx * window_latent_frames
            end = start + window_latent_frames
            dev = pipeline_self._execution_device
            dtype = pipeline_self.transformer.dtype

            ctrl_chunk = _vace_latents[:, :, start:end].to(device=dev, dtype=dtype)

            extra: dict = {
                "control_hidden_states": ctrl_chunk,
                "control_hidden_states_scale": _scale,
                "vace_last_chunk_padding": vace_last_chunk_padding,
                "vace_only_inject_at_full_resolution": vace_only_inject_at_full_resolution,
                "vace_process_at_full_resolution": vace_process_at_full_resolution,
                "inject_hints_to_history": inject_hints_to_history,
            }

            # Feature 2: provide short-history VACE control latents.
            # Short history covers the `history_sizes[-1]` most-recent latent frames
            # immediately preceding the current chunk (plus, when keep_first_frame=True,
            # the very first generated frame as a prefix).  We slice those from
            # vace_latents_full using the same logic as the pipeline's history buffer.
            if inject_hints_to_history:
                hist_end = start                 # = chunk_idx * window_latent_frames
                hist_short_size = history_sizes[-1]
                hist_short_start = max(0, hist_end - hist_short_size)
                ctrl_hist_short = _vace_latents[:, :, hist_short_start:hist_end].to(
                    device=dev, dtype=dtype
                )
                if keep_first_frame and _vace_latents.shape[2] > 0:
                    ctrl_prefix = _vace_latents[:, :, 0:1].to(device=dev, dtype=dtype)
                    ctrl_hist_short = torch.cat([ctrl_prefix, ctrl_hist_short], dim=2)

                extra["control_hidden_states_history_short"] = ctrl_hist_short

            return extra

        def _stage1_with_vace(self_inner, *args, **kwargs):
            chunk_idx = _chunk_counter[0]
            _chunk_counter[0] += 1
            kwargs["extra_transformer_kwargs"] = _build_vace_kwargs(chunk_idx)
            return _orig_stage1(self_inner, *args, **kwargs)

        def _stage2_with_vace(self_inner, *args, **kwargs):
            # stage2 is called once per chunk (all pyramid stages inside it)
            chunk_idx = _chunk_counter[0]
            _chunk_counter[0] += 1
            kwargs["extra_transformer_kwargs"] = _build_vace_kwargs(chunk_idx)
            return _orig_stage2(self_inner, *args, **kwargs)

        # Bind the patched methods for this call
        import types
        self.stage1_sample = types.MethodType(_stage1_with_vace, self)
        self.stage2_sample = types.MethodType(_stage2_with_vace, self)

        try:
            result = super().__call__(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                num_videos_per_prompt=num_videos_per_prompt,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                output_type=output_type,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
                image=image,
                image_latents=image_latents,
                fake_image_latents=fake_image_latents,
                add_noise_to_image_latents=add_noise_to_image_latents,
                image_noise_sigma_min=image_noise_sigma_min,
                image_noise_sigma_max=image_noise_sigma_max,
                video=video,
                video_latents=video_latents,
                add_noise_to_video_latents=add_noise_to_video_latents,
                video_noise_sigma_min=video_noise_sigma_min,
                video_noise_sigma_max=video_noise_sigma_max,
                use_interpolate_prompt=use_interpolate_prompt,
                interpolate_time_list=interpolate_time_list,
                interpolation_steps=interpolation_steps,
                history_sizes=history_sizes,
                num_latent_frames_per_chunk=num_latent_frames_per_chunk,
                keep_first_frame=keep_first_frame,
                is_skip_first_chunk=is_skip_first_chunk,
                is_enable_stage2=is_enable_stage2,
                pyramid_num_stages=pyramid_num_stages,
                pyramid_num_inference_steps_list=pyramid_num_inference_steps_list,
                use_zero_init=use_zero_init,
                zero_steps=zero_steps,
                is_amplify_first_chunk=is_amplify_first_chunk,
            )
        finally:
            # Restore original unbound methods
            del self.stage1_sample
            del self.stage2_sample

        return result
