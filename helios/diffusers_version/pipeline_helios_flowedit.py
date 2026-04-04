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
        latents_history_short_target=None,
        latents_history_mid_target=None,
        latents_history_long_target=None,
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
        batch_size = latents.shape[0]

        # Pre-cast history buffers once (avoid repeated .to() inside the inner loop).
        lh_short_src = latents_history_short.to(torch.float32)
        lh_mid_src   = latents_history_mid.to(torch.float32)
        lh_long_src  = latents_history_long.to(torch.float32)
        if latents_history_short_target is None:
            lh_short_tar = lh_short_src
            lh_mid_tar = lh_mid_src
            lh_long_tar = lh_long_src
        else:
            lh_short_tar = latents_history_short_target.to(torch.float32)
            lh_mid_tar   = latents_history_mid_target.to(torch.float32)
            lh_long_tar  = latents_history_long_target.to(torch.float32)

        patch_size = self.transformer.config.patch_size

        for i_s in range(pyramid_num_stages):
            # ── Per-stage start_point / end_point (mirrors compute_sdedit_input) ─
            # Keep pyramid tensors on GPU temporarily; free them after Zt_src is computed.
            x0_k    = latent_pyramid[i_s].to(device=device, dtype=torch.float32)
            noise_k = noise_pyramid[i_s].to(device=device, dtype=torch.float32)
            x0_prev = None
            x0_prev_up = None
            if i_s == 0:
                start_point = noise_k
            else:
                x0_prev = latent_pyramid[i_s - 1].to(device=device, dtype=torch.float32)
                tgt_h, tgt_w = x0_k.shape[-2], x0_k.shape[-1]
                _Bp, _Cp, _Tp = x0_prev.shape[:3]
                _flat = x0_prev.permute(0,2,1,3,4).reshape(_Bp*_Tp, _Cp, x0_prev.shape[-2], x0_prev.shape[-1])
                x0_prev_up = F.interpolate(_flat, size=(tgt_h, tgt_w), mode="nearest").reshape(_Bp,_Tp,_Cp,tgt_h,tgt_w).permute(0,2,1,3,4)
                del _flat  # keep x0_prev alive for delta computation in inter-stage block
                start_sigma_k = self.scheduler.start_sigmas[i_s]
                start_point = start_sigma_k * noise_k + (1 - start_sigma_k) * x0_prev_up

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

            # Inter-stage: carry the EDIT DELTA (Zt_edit - x0_src_coarse) to the next resolution.
            # Bilinear upsample of the *correction* (not raw latents) avoids block artifacts
            # and anchors Zt_edit to the fine-res clean source, making Zt_tar = delta_up + Zt_src.
            if i_s > flowedit_start_stage:
                _fe_B, _fe_C, _fe_T, _h, _w = latents.shape
                _tgt_h, _tgt_w = _h * 2, _w * 2
                delta_edit = latents.float() - x0_prev.float()   # edit correction at coarse stage
                _flat = delta_edit.permute(0,2,1,3,4).reshape(_fe_B*_fe_T, _fe_C, _h, _w)
                delta_up = (F.interpolate(_flat, size=(_tgt_h, _tgt_w), mode="bilinear")
                            .reshape(_fe_B, _fe_T, _fe_C, _tgt_h, _tgt_w)
                            .permute(0, 2, 1, 3, 4))
                del _flat, delta_edit
                latents = (x0_k + delta_up).to(torch.float32)   # anchor to fine-res source + edit
                del delta_up
            if x0_prev is not None:
                del x0_prev

            # Reset transformer's stateful KV cache at each stage boundary — caches from
            # the previous stage's resolution don't apply here and take significant VRAM.
            self.transformer._reset_stateful_cache(recurse=True)
            gc.collect()
            torch.cuda.empty_cache()

            # Define transformer call once per stage (captures constant stage-level state).
            @torch.no_grad()
            def _fe_call(hidden, embeds, ctx_key, _ts=None, history_role="src"):
                gc.collect()
                torch.cuda.empty_cache()
                if history_role == "tar":
                    hist_short = lh_short_tar
                    hist_mid = lh_mid_tar
                    hist_long = lh_long_tar
                else:
                    hist_short = lh_short_src
                    hist_mid = lh_mid_src
                    hist_long = lh_long_src
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
                        latents_history_short=hist_short.to(transformer_dtype),
                        latents_history_mid=hist_mid.to(transformer_dtype),
                        latents_history_long=hist_long.to(transformer_dtype),
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
                    # After delta-based inter-stage init, latents = x0_k + delta_up, so:
                    #   Zt_tar = latents + Zt_src - x0_k = delta_up + Zt_src
                    # This is the pure FlowEdit formula: edit delta + source trajectory.
                    Zt_src = (sigma_t * start_point + (1 - sigma_t) * end_point).to(torch.float32)
                    Zt_tar = (latents.to(torch.float32) + Zt_src - x0_k.to(torch.float32))

                    if flowedit_src_neg_embeds is not None:
                        # FlowEdit: 2-4 calls; delete velocity intermediates ASAP.
                        v_tar_cond = _fe_call(Zt_tar, prompt_embeds, "fe_tar_cond", timestep, history_role="tar").float()
                        if guidance_scale > 1:
                            v_tar_uncond = _fe_call(
                                Zt_tar, negative_prompt_embeds, "fe_tar_uncond", timestep, history_role="tar"
                            ).float()
                            Vt_tar = v_tar_uncond + guidance_scale * (v_tar_cond - v_tar_uncond)
                            del v_tar_cond, v_tar_uncond
                        else:
                            Vt_tar = v_tar_cond
                            del v_tar_cond
                        del Zt_tar
                        v_src_cond = _fe_call(Zt_src, flowedit_src_pos_embeds, "fe_src_cond", timestep, history_role="src").float()
                        if flowedit_src_gs > 1:
                            v_src_uncond = _fe_call(
                                Zt_src, flowedit_src_neg_embeds, "fe_src_uncond", timestep, history_role="src"
                            ).float()
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
                        vq     = _fe_call(Zt_src, flowedit_src_pos_embeds, "fa_vq",     timestep, history_role="src").float()
                        vp_tar = _fe_call(Zt_tar, prompt_embeds,           "fa_vp_tar", timestep, history_role="tar").float()
                        del Zt_tar
                        if guidance_scale > 1:
                            vp_src = _fe_call(
                                Zt_src, flowedit_src_pos_embeds, "fa_vp_src", timestep, history_role="src"
                            ).float()
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
