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

import gc
import math
import numpy as np
import torch
import torch.nn.functional as F

from diffusers.utils import is_torch_xla_available

from .pipeline_helios_diffusers import HeliosPipeline, calculate_shift

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class HeliosPipelineFlowEdit(HeliosPipeline):
    """HeliosPipeline with FlowEdit / FlowAlign support via stage2_sample."""

    def stage2_sample(
        self,
        latents,                         # full-res clean X0_src (will be downsampled internally)
        pyramid_num_stages,
        pyramid_num_inference_steps_list,
        prompt_embeds,                   # target positive embeds
        negative_prompt_embeds,          # target negative embeds
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
        use_zero_init=True,              # unused, kept for API compat
        zero_steps=1,                    # unused, kept for API compat
        is_amplify_first_chunk=False,
        # FlowEdit / FlowAlign specific
        flowedit_X0_src=None,            # same as latents (passed explicitly for clarity)
        flowedit_src_pos_embeds=None,    # source positive embeds
        flowedit_src_neg_embeds=None,    # None -> FlowAlign (3-call); provided -> FlowEdit (4-call)
        flowedit_src_gs=1.0,             # CFG scale for source velocity
        flowedit_start_stage=0,          # edit_stage floor (same convention as SDEdit)
        flowedit_start_step=0,           # first step in start_stage to apply edit
        flowedit_zeta=0.0,               # 0=FlowEdit; >0=FlowAlign DIFS term
        base_seed=None,                  # unused, kept for API compat
        chunk_frame_offset=0,            # unused, kept for API compat
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=["latents"],
        progress_bar=None,
        extra_transformer_kwargs=None,
    ):
        batch_size, num_channel, num_frames, height, width = latents.shape
        extra = extra_transformer_kwargs or {}

        # ── Build pyramids (once, before stage loop) ──────────────────────────────
        # latent_pyramid[k]: clean source downsampled to stage k resolution
        #   (bilinear, no ×2 — signal not noise).
        # noise_pyramid[k]: noise downsampled to stage k (bilinear + ×2 per level)
        #   matching the training noise_list in prepare_stage2_clean_input.
        x0_full = flowedit_X0_src.to(device=device, dtype=torch.float32)
        _B, _C, _T = x0_full.shape[:3]

        latent_pyramid = [x0_full]   # [0]=full, grows → [0]=full, [-1]=coarsest; reversed below
        noise_full = torch.randn(x0_full.shape, device=device, dtype=torch.float32, generator=generator)
        noise_pyramid  = [noise_full]
        _cur_lat = x0_full
        _cur_noi = noise_full
        _h, _w = height, width
        for _ in range(pyramid_num_stages - 1):
            _h //= 2; _w //= 2
            _flat = _cur_lat.permute(0,2,1,3,4).reshape(_B*_T, _C, _cur_lat.shape[-2], _cur_lat.shape[-1])
            _cur_lat = F.interpolate(_flat, size=(_h, _w), mode="bilinear").reshape(_B,_T,_C,_h,_w).permute(0,2,1,3,4)
            latent_pyramid.append(_cur_lat)
            _flat = _cur_noi.permute(0,2,1,3,4).reshape(_B*_T, _C, _cur_noi.shape[-2], _cur_noi.shape[-1])
            _cur_noi = (F.interpolate(_flat, size=(_h, _w), mode="bilinear") * 2).reshape(_B,_T,_C,_h,_w).permute(0,2,1,3,4)
            noise_pyramid.append(_cur_noi)
        # reverse so index 0 = coarsest, index pyramid_num_stages-1 = full res
        latent_pyramid = list(reversed(latent_pyramid))
        noise_pyramid  = list(reversed(noise_pyramid))
        # Keep pyramids on CPU to avoid fragmenting GPU while transformer is active.
        torch.cuda.empty_cache()

        # Zt_edit initialised to clean source at the entry stage.
        latents = latent_pyramid[flowedit_start_stage].to(device=device)
        anchor_latents = latents
        batch_size = latents.shape[0]

        # Pre-cast history buffers once (avoid repeated .to() inside the inner loop).
        lh_short = latents_history_short.to(torch.float32)
        lh_mid   = latents_history_mid.to(torch.float32)
        lh_long  = latents_history_long.to(torch.float32)

        patch_size = self.transformer.config.patch_size

        for i_s in range(pyramid_num_stages):
            # ── Per-stage start_point / end_point (mirrors compute_sdedit_input) ─
            # Keep pyramid tensors on GPU temporarily; free them after Zt_src is computed.
            x0_k    = latent_pyramid[i_s].to(device=device, dtype=torch.float32)
            noise_k = noise_pyramid[i_s].to(device=device, dtype=torch.float32)
            x0_prev_up = None
            if i_s == 0:
                start_point = noise_k
            else:
                x0_prev = latent_pyramid[i_s - 1].to(device=device, dtype=torch.float32)
                tgt_h, tgt_w = x0_k.shape[-2], x0_k.shape[-1]
                _Bp, _Cp, _Tp = x0_prev.shape[:3]
                _flat = x0_prev.permute(0,2,1,3,4).reshape(_Bp*_Tp, _Cp, x0_prev.shape[-2], x0_prev.shape[-1])
                x0_prev_up = F.interpolate(_flat, size=(tgt_h, tgt_w), mode="nearest").reshape(_Bp,_Tp,_Cp,tgt_h,tgt_w).permute(0,2,1,3,4)
                del x0_prev, _flat
                start_sigma_k = self.scheduler.start_sigmas[i_s]
                start_point = start_sigma_k * noise_k + (1 - start_sigma_k) * x0_prev_up
                # del x0_prev_up

            if i_s < pyramid_num_stages - 1:
                end_sigma_k = self.scheduler.end_sigmas[i_s]
                end_point = end_sigma_k * noise_k + (1 - end_sigma_k) * x0_k
            else:
                end_point = x0_k   # last stage: clean source is the target
            del noise_k

            # Compute dynamic time-shift mu from current latent spatial size.
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

            self.scheduler.set_timesteps(
                pyramid_num_inference_steps_list[i_s],
                i_s,
                device=device,
                mu=mu,
                is_amplify_first_chunk=is_amplify_first_chunk,
            )
            timesteps = self.scheduler.timesteps

            # Inter-stage: bilinear upsample Zt_edit (signal — no DMD re-noise for FlowEdit).
            if i_s > flowedit_start_stage:
                # _fe_B, _fe_C, _fe_T, _h, _w = latents.shape
                # _tgt_h, _tgt_w = _h * 2, _w * 2
                # _flat = latents.float().permute(0,2,1,3,4).reshape(_fe_B*_fe_T, _fe_C, _h, _w)
                # latents = (F.interpolate(_flat, size=(_tgt_h, _tgt_w), mode="bilinear")
                #            .reshape(_fe_B, _fe_T, _fe_C, _tgt_h, _tgt_w)
                #            .permute(0, 2, 1, 3, 4).to(torch.float32))
                # del _flat

                num_frames = latents.shape[2]
                batch_size, num_channel, num_frames, _h, _w = latents.shape
                height, width = _h * 2, _w * 2

                latents = latents.permute(0, 2, 1, 3, 4).reshape(
                    batch_size * num_frames, num_channel, _h, _w
                )
                latents = F.interpolate(latents, size=(height, width), mode="nearest")
                latents = latents.reshape(batch_size, num_frames, num_channel, height, width).permute(0, 2, 1, 3, 4)
                anchor_latents = latents
                
                # # Re-noise to fix block artifacts at upsampled resolution.
                # ori_sigma = 1 - self.scheduler.ori_start_sigmas[i_s]
                # gamma = self.scheduler.config.gamma
                # alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                # beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

                # batch_size, num_channel, num_frames, height, width = latents.shape
                # if base_seed is not None:
                #     _block_gen = torch.Generator(device=device).manual_seed(base_seed + chunk_frame_offset + i_s)
                # else:
                #     _block_gen = generator
                # noise = self.sample_block_noise(
                #     batch_size, num_channel, num_frames, height, width, patch_size, device, _block_gen
                # )
                # noise = noise.to(device=device, dtype=torch.float32)
                # latents = alpha * latents + beta * noise

            # Reset transformer's stateful KV cache at each stage boundary — caches from
            # the previous stage's resolution don't apply here and take significant VRAM.
            self.transformer._reset_stateful_cache(recurse=True)
            gc.collect()
            torch.cuda.empty_cache()

            # Define transformer call once per stage (captures constant stage-level state).
            @torch.no_grad()
            def _fe_call(hidden, embeds, ctx_key, _ts=None):
                gc.collect()
                torch.cuda.empty_cache()
                with self.transformer.cache_context(ctx_key):
                    return self.transformer(
                        hidden_states=hidden.to(transformer_dtype),
                        timestep=_ts,
                        encoder_hidden_states=embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=indices_latents_history_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_history_long,
                        latents_history_short=lh_short.to(transformer_dtype),
                        latents_history_mid=lh_mid.to(transformer_dtype),
                        latents_history_long=lh_long.to(transformer_dtype),
                        **extra,
                    )[0]

            for idx, t in enumerate(timesteps):
                timestep = t.expand(latents.shape[0]).to(torch.int64)

                sigma_t    = float(self.scheduler.sigmas[idx])
                sigma_next = float(self.scheduler.sigmas[idx + 1])

                # Apply edit only from (flowedit_start_stage, flowedit_start_step) onward.
                # Stages before start_stage: Zt_edit stays at x0_src (no velocity update).
                # Entry stage: skip the first flowedit_start_step steps.
                _do_edit = (i_s > flowedit_start_stage) or (
                    i_s == flowedit_start_stage and idx >= flowedit_start_step
                )

                if _do_edit:
                    # Stage2-consistent forward process (same formula as training):
                    #   Zt_src = sigma_t * start_point + (1 - sigma_t) * end_point
                    # Zt_tar replaces x0_k with Zt_edit (the running edited latent).
                    Zt_src = (sigma_t * start_point + (1 - sigma_t) * end_point).to(torch.float32) # [B, C, T, H, W]
                    # x0_tar = x0_prev_up * (1 - x0_tar_ratio[idx]) + x0_k * x0_tar_ratio[idx]
                    x0_tar = sigma_t * x0_prev_up + (1 - sigma_t) * x0_k
                    # if idx == 0 and i_s > flowedit_start_stage:
                    #     Zt_tar = (latents.to(torch.float32)) # + x0_prev_up - x0_k.to(torch.float32))
                    # else:   
                    #     Zt_tar = (latents.to(torch.float32) + Zt_src - x0_tar.to(torch.float32))
                    Zt_tar = (latents.to(torch.float32) + Zt_src - x0_tar.to(torch.float32))

                    if idx == 0 and i_s > flowedit_start_stage:
                        # Re-noise to fix block artifacts at upsampled resolution.
                        ori_sigma = 1 - self.scheduler.ori_start_sigmas[i_s]
                        gamma = self.scheduler.config.gamma
                        alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                        beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

                        batch_size, num_channel, num_frames, height, width = Zt_tar.shape
                        if base_seed is not None:
                            _block_gen = torch.Generator(device=device).manual_seed(base_seed + chunk_frame_offset + i_s)
                        else:
                            _block_gen = generator
                        noise = self.sample_block_noise(
                            batch_size, num_channel, num_frames, height, width, patch_size, device, _block_gen
                        )
                        noise = noise.to(device=device, dtype=torch.float32)
                        # Zt_src = alpha * Zt_src + beta * noise
                        Zt_tar = alpha * Zt_tar + beta * noise

                    if flowedit_src_neg_embeds is not None:
                        # FlowEdit: 2-4 calls; delete velocity intermediates ASAP.
                        v_tar_cond = _fe_call(Zt_tar, prompt_embeds, "fe_tar_cond", timestep).float()
                        if guidance_scale > 1:
                            v_tar_uncond = _fe_call(Zt_tar, negative_prompt_embeds, "fe_tar_uncond", timestep).float()
                            Vt_tar = v_tar_uncond + guidance_scale * (v_tar_cond - v_tar_uncond)
                            del v_tar_cond, v_tar_uncond
                        else:
                            Vt_tar = v_tar_cond
                            del v_tar_cond
                        del Zt_tar
                        v_src_cond = _fe_call(Zt_src, flowedit_src_pos_embeds, "fe_src_cond", timestep).float()
                        if flowedit_src_gs > 1:
                            v_src_uncond = _fe_call(Zt_src, flowedit_src_neg_embeds, "fe_src_uncond", timestep).float()
                            Vt_src = v_src_uncond + flowedit_src_gs * (v_src_cond - v_src_uncond)
                            del v_src_cond, v_src_uncond
                        else:
                            Vt_src = v_src_cond
                            del v_src_cond
                        del Zt_src
                        velocity = Vt_tar - Vt_src
                        del Vt_tar, Vt_src
                    else:
                        # FlowAlign: 3-call classifier guidance (no source uncond).
                        vq     = _fe_call(Zt_src, flowedit_src_pos_embeds, "fa_vq",     timestep).float()
                        vp_tar = _fe_call(Zt_tar, prompt_embeds,           "fa_vp_tar", timestep).float()
                        del Zt_tar
                        if guidance_scale > 1:
                            vp_src = _fe_call(Zt_src, flowedit_src_pos_embeds, "fa_vp_src", timestep).float()
                            vp = vp_src + guidance_scale * (vp_tar - vp_src)
                            del vp_src
                        else:
                            vp = vp_tar
                        del vp_tar
                        difs = None
                        if flowedit_zeta > 0:
                            difs = (Zt_src - sigma_t * vq) - (latents.float() + Zt_src - x0_k.float() - sigma_t * vp)
                        del Zt_src
                        velocity = vp - vq
                        del vp, vq

                    dt = sigma_next - sigma_t  # negative (sigma decreasing)
                    if flowedit_zeta > 0 and difs is not None:
                        latents = (latents.float() + dt * velocity + flowedit_zeta * difs).to(torch.float32)
                        del difs
                    else:
                        latents = (latents.float() + dt * velocity).to(torch.float32)
                    del velocity

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, idx, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        return latents
