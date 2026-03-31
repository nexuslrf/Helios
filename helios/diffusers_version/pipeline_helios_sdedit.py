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

import math
from typing import Callable

import torch
import torch.nn.functional as F

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.utils import is_torch_xla_available

from .pipeline_helios_diffusers import HeliosPipeline, calculate_shift, optimized_scale

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class HeliosPipelineSDEdit(HeliosPipeline):
    """HeliosPipeline with pyramid-aware SDEdit support via stage2_sample."""

    def stage2_sample(
        self,
        latents,                         # pre-computed x_t at sdedit_start_stage resolution
        pyramid_num_stages,
        pyramid_num_inference_steps_list,
        prompt_embeds,
        negative_prompt_embeds,
        guidance_scale=1.0,
        indices_hidden_states=None,
        indices_latents_history_short=None,
        indices_latents_history_mid=None,
        indices_latents_history_long=None,
        latents_history_short=None,
        latents_history_mid=None,
        latents_history_long=None,
        attention_kwargs=None,
        device=None,
        transformer_dtype=None,
        generator=None,
        use_zero_init=True,
        zero_steps=1,
        is_amplify_first_chunk=False,
        # SDEdit-specific
        sdedit_start_stage=0,
        sdedit_start_step=0,
        sdedit_noise_anchor=None,        # training-consistent DMD anchor at entry stage
        base_seed=None,
        chunk_frame_offset=0,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=["latents"],
        progress_bar=None,
        extra_transformer_kwargs=None,
    ):
        # latents is already at sdedit_start_stage resolution — no downsampling needed.
        batch_size, num_channel, num_frames, height, width = latents.shape
        extra = extra_transformer_kwargs or {}
        start_point_list = [sdedit_noise_anchor] if self.config.is_distilled else None

        i = 0
        for i_s in range(pyramid_num_stages):
            # Skip pyramid stages before the SDEdit entry stage.
            if i_s < sdedit_start_stage:
                continue

            # Compute dynamic time-shift mu from current latent spatial size.
            patch_size = self.transformer.config.patch_size
            image_seq_len = (latents.shape[-1] * latents.shape[-2] * latents.shape[-3]) // (
                patch_size[0] * patch_size[1] * patch_size[2]
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )

            # Truncate sigma schedule for the SDEdit entry stage when starting mid-stage.
            steps_this_stage = pyramid_num_inference_steps_list[i_s]
            stage_start_sigma = None
            if i_s == sdedit_start_stage and sdedit_start_step > 0:
                steps_this_stage = max(1, steps_this_stage - sdedit_start_step)
                T_full = pyramid_num_inference_steps_list[i_s]
                s0 = float(self.scheduler.sigmas_per_stage[i_s][0].item())
                s1 = float(self.scheduler.sigmas_per_stage[i_s][-1].item())
                # sigma corresponding to sdedit_start_step in the full T-step linspace
                stage_start_sigma = s0 + (s1 - s0) * sdedit_start_step / max(T_full - 1, 1)

            self.scheduler.set_timesteps(
                steps_this_stage,
                i_s,
                device=device,
                mu=mu,
                is_amplify_first_chunk=is_amplify_first_chunk,
                start_sigma=stage_start_sigma,
            )
            timesteps = self.scheduler.timesteps

            # Inter-stage block: runs for all i_s > sdedit_start_stage.
            # The entry stage uses the pre-computed x_t directly, so no inter-stage processing there.
            if i_s > sdedit_start_stage:
                height *= 2
                width *= 2
                num_frames = latents.shape[2]
                batch_size, num_channel, num_frames, _h, _w = latents.shape

                latents = latents.permute(0, 2, 1, 3, 4).reshape(
                    batch_size * num_frames, num_channel, _h, _w
                )
                latents = F.interpolate(latents, size=(height, width), mode="nearest")
                latents = latents.reshape(batch_size, num_frames, num_channel, height, width).permute(0, 2, 1, 3, 4)

                # Re-noise to fix block artifacts at upsampled resolution.
                ori_sigma = 1 - self.scheduler.ori_start_sigmas[i_s]
                gamma = self.scheduler.config.gamma
                alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

                batch_size, num_channel, num_frames, height, width = latents.shape
                if base_seed is not None:
                    _block_gen = torch.Generator(device=device).manual_seed(base_seed + chunk_frame_offset + i_s)
                else:
                    _block_gen = generator
                noise = self.sample_block_noise(
                    batch_size, num_channel, num_frames, height, width, patch_size, device, _block_gen
                )
                noise = noise.to(device=device, dtype=torch.float32)
                latents = alpha * latents + beta * noise

                if self.config.is_distilled:
                    start_point_list.append(latents)

            for idx, t in enumerate(timesteps):
                timestep = t.expand(latents.shape[0]).to(torch.int64)

                # Standard transformer call (SDEdit uses the same denoising as normal generation).
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents.to(transformer_dtype),
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=indices_latents_history_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_history_long,
                        latents_history_short=latents_history_short.to(transformer_dtype),
                        latents_history_mid=latents_history_mid.to(transformer_dtype),
                        latents_history_long=latents_history_long.to(transformer_dtype),
                        **extra,
                    )[0]

                if self.do_classifier_free_guidance:
                    with self.transformer.cache_context("uncond"):
                        noise_uncond = self.transformer(
                            hidden_states=latents.to(transformer_dtype),
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                            indices_hidden_states=indices_hidden_states,
                            indices_latents_history_short=indices_latents_history_short,
                            indices_latents_history_mid=indices_latents_history_mid,
                            indices_latents_history_long=indices_latents_history_long,
                            latents_history_short=latents_history_short.to(transformer_dtype),
                            latents_history_mid=latents_history_mid.to(transformer_dtype),
                            latents_history_long=latents_history_long.to(transformer_dtype),
                            **extra,
                        )[0]

                    if self.config.is_cfg_zero_star:
                        noise_pred_text = noise_pred
                        positive_flat = noise_pred_text.view(batch_size, -1)
                        negative_flat = noise_uncond.view(batch_size, -1)

                        alpha_scale = optimized_scale(positive_flat, negative_flat)
                        alpha_scale = alpha_scale.view(batch_size, *([1] * (len(noise_pred_text.shape) - 1)))
                        alpha_scale = alpha_scale.to(noise_pred_text.dtype)

                        if (i_s == sdedit_start_stage and idx <= zero_steps) and use_zero_init:
                            noise_pred = noise_pred_text * 0.0
                        else:
                            noise_pred = noise_uncond * alpha_scale + guidance_scale * (
                                noise_pred_text - noise_uncond * alpha_scale
                            )
                    else:
                        noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                # start_point_list is indexed from 0 = sdedit_start_stage.
                spl_idx = i_s - sdedit_start_stage
                latents = self.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    generator=generator,
                    return_dict=False,
                    cur_sampling_step=idx,
                    dmd_noisy_tensor=start_point_list[spl_idx] if start_point_list is not None else None,
                    dmd_sigmas=self.scheduler.sigmas,
                    dmd_timesteps=self.scheduler.timesteps,
                    all_timesteps=timesteps,
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

                i += 1

        return latents
