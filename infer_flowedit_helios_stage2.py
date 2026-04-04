"""
FlowEdit and FlowAlign inference for Helios using the stage2 DMD pyramid pipeline.

Like infer_sdedit_helios_stage2.py, this script delegates all pyramid stage management
(sigma schedules, inter-stage transitions, history, timestep conditioning) to stage2_sample.
FlowEdit mode is triggered by passing flowedit_X0_src to stage2_sample.

Algorithm (per chunk, per stage, per step at/after edit_stage entry point):

  FlowEdit (--edit_type flowedit):
    Zt_src = (1 - σ) * X0_src_stage + σ * ε
    Zt_tar = Zt_edit + Zt_src - X0_src_stage
    Vt_tar = uncond_tar + gs_tar * (cond_tar - uncond_tar)   [2 transformer calls on Zt_tar]
    Vt_src = uncond_src + gs_src * (cond_src - uncond_src)   [2 transformer calls on Zt_src]
    Zt_edit += dt * (Vt_tar - Vt_src)                        [pure Euler, no DMD]

  FlowAlign (--edit_type flowalign):
    Zt_src, Zt_tar as above
    vq     = transformer(Zt_src, src_prompt)                  [1 call]
    vp_src = transformer(Zt_tar, src_prompt)                  [1 call]
    vp_tar = transformer(Zt_tar, tar_prompt)                  [1 call]
    vp = vp_src + gs_tar * (vp_tar - vp_src)
    Zt_edit += dt * (vp - vq) + zeta * [(Zt_src - σ*vq) - (Zt_tar - σ*vp)]

stage2_sample handles:
  - Downsample X0_src to coarsest pyramid stage at init
  - Bilinear upsample of Zt_edit and X0_src_stage between stages (no gamma renoise)
  - Correct sigma schedule and timestep conditioning per stage (set_timesteps)
  - History buffer for causal chunk conditioning
"""

import importlib
import os
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

if importlib.util.find_spec("torch_npu") is not None:
    import torch_npu
else:
    torch_npu = None

from helios.diffusers_version.pipeline_helios_flowedit import HeliosPipelineFlowEdit
from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler
from helios.diffusers_version.transformer_helios_diffusers import HeliosTransformer3DModel
from helios.modules.helios_kernels import (
    replace_all_norms_with_flash_norms,
    replace_rmsnorm_with_fp32,
    replace_rope_with_flash_rope,
)

from diffusers.models import AutoencoderKLWan
from diffusers.utils import export_to_video, load_video
from diffusers.utils.torch_utils import randn_tensor


def parse_args():
    parser = argparse.ArgumentParser(description="FlowEdit / FlowAlign (stage2 pyramid DMD) video editing with Helios")

    parser.add_argument("--base_model_path", type=str, default="BestWishYsh/Helios-Distilled")
    parser.add_argument("--transformer_path", type=str, default="BestWishYsh/Helios-Distilled")
    parser.add_argument("--output_folder", type=str, default="./output_helios/flowedit_stage2")

    parser.add_argument("--edit_type", type=str, default="flowedit", choices=["flowedit", "flowalign"],
                        help="flowedit: 4-call CFG differential; flowalign: 3-call DIFS alignment")

    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--source_prompt", type=str, required=True)
    parser.add_argument("--target_prompt", type=str, required=True)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
                "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
                "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
                "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
                "in the background, walking backwards",
    )
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=33)

    # Edit hyperparams
    parser.add_argument(
        "--edit_stage", type=float, default=1.0,
        help="Pyramid entry coordinate: X = stage_idx + step_frac.  "
             "Stages < floor(X) pass X0_src through unchanged (no FlowEdit).  "
             "E.g. 0.0=max edit, 1.0=enter at stage 1 step 0, 2.0=min edit (fine details only).")
    parser.add_argument("--source_guidance_scale", type=float, default=1.0,
                        help="FlowEdit: CFG scale for source velocity (1.0 = no CFG)")
    parser.add_argument("--target_guidance_scale", type=float, default=1.0,
                        help="FlowEdit/FlowAlign: scale for target velocity (1.0 = no CFG; stage2 is distilled)")
    parser.add_argument("--zeta_scale", type=float, default=1e-3,
                        help="FlowAlign: DIFS alignment term weight")

    parser.add_argument(
        "--pyramid_num_inference_steps_list",
        type=int,
        nargs="+",
        default=[10, 10, 10],
    )
    parser.add_argument("--pyramid_num_stages", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_latent_frames_per_chunk", type=int, default=9)
    parser.add_argument("--weight_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def main():
    args = parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    weight_dtype = dtype_map[args.weight_dtype]

    os.makedirs(args.output_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # ── Load models ──────────────────────────────────────────────────────────
    transformer = HeliosTransformer3DModel.from_pretrained(
        args.transformer_path, subfolder="transformer", torch_dtype=weight_dtype,
    )
    transformer = replace_rmsnorm_with_fp32(transformer)
    transformer = replace_all_norms_with_flash_norms(transformer)
    replace_rope_with_flash_rope()

    cuda_major = torch.cuda.get_device_capability()[0]
    try:
        transformer.set_attention_backend("_flash_3" if cuda_major >= 9 else "flash")
    except Exception:
        transformer.set_attention_backend("flash")

    vae = AutoencoderKLWan.from_pretrained(args.base_model_path, subfolder="vae", torch_dtype=torch.float32)
    scheduler = HeliosScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")
    pipe = HeliosPipelineFlowEdit.from_pretrained(
        args.base_model_path, transformer=transformer, vae=vae, scheduler=scheduler,
        torch_dtype=weight_dtype,
    ).to(device)

    transformer_dtype = transformer.dtype

    # ── Load and encode source video first (before prompts, to avoid VRAM fragmentation) ──
    print(f"Loading source video: {args.video_path}")
    source_video = load_video(args.video_path)

    video_tensor = pipe.video_processor.preprocess_video(source_video, height=args.height, width=args.width)
    video_tensor = video_tensor.to(dtype=vae.dtype)

    latents_mean = (
        torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, vae.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, vae.dtype)

    num_latent_frames_per_chunk = args.num_latent_frames_per_chunk
    vae_temporal = pipe.vae_scale_factor_temporal   # 4
    min_frames = (num_latent_frames_per_chunk - 1) * vae_temporal + 1  # 33

    num_video_frames = video_tensor.shape[2]
    num_chunks = max(1, num_video_frames // min_frames)
    total_valid_frames = num_chunks * min_frames
    start_frame = num_video_frames - total_valid_frames

    print(f"Video: {num_video_frames} frames → {num_chunks} chunks of {min_frames} frames each")
    print("Encoding source video to latents...")

    X0_src = []
    latents_mean_cpu, latents_std_cpu = latents_mean.cpu(), latents_std.cpu()
    with torch.no_grad():
        for k in range(num_chunks):
            chunk_start = start_frame + k * min_frames
            chunk_end = chunk_start + min_frames
            chunk = video_tensor[:, :, chunk_start:chunk_end]
            z = vae.encode(chunk.to(device=device)).latent_dist.sample(generator=generator).cpu()
            z = (z - latents_mean_cpu) * latents_std_cpu
            X0_src.append(z.to(torch.float32))
    del video_tensor
    torch.cuda.empty_cache()

    # ── Encode prompts (after video encoding to avoid VRAM fragmentation) ────
    print("Encoding prompts...")
    src_pos_embed, src_neg_embed = pipe.encode_prompt(
        prompt=args.source_prompt,
        negative_prompt=args.negative_prompt,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=device,
    )
    tar_pos_embed, tar_neg_embed = pipe.encode_prompt(
        prompt=args.target_prompt,
        negative_prompt=args.negative_prompt,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=device,
    )
    src_pos_embed = src_pos_embed.to(transformer_dtype)
    src_neg_embed = src_neg_embed.to(transformer_dtype)
    tar_pos_embed = tar_pos_embed.to(transformer_dtype)
    tar_neg_embed = tar_neg_embed.to(transformer_dtype)

    # ── Positional indices (same as infer_sdedit_helios_stage2) ──────────────
    history_sizes = sorted([16, 2, 1], reverse=True)  # [16, 2, 1]
    num_history_latent_frames = sum(history_sizes)
    keep_first_frame = True

    indices = torch.arange(0, sum([1, *history_sizes, num_latent_frames_per_chunk]))
    (
        indices_prefix,
        indices_latents_history_long,
        indices_latents_history_mid,
        indices_latents_history_1x,
        indices_hidden_states,
    ) = indices.split([1, *history_sizes, num_latent_frames_per_chunk], dim=0)
    indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)

    indices_hidden_states = indices_hidden_states.unsqueeze(0)
    indices_latents_history_short = indices_latents_history_short.unsqueeze(0)
    indices_latents_history_mid = indices_latents_history_mid.unsqueeze(0)
    indices_latents_history_long = indices_latents_history_long.unsqueeze(0)

    x0_ref = X0_src[0]
    batch_size, C = x0_ref.shape[0], x0_ref.shape[1]
    H_lat, W_lat = x0_ref.shape[-2], x0_ref.shape[-1]

    # Keep separate history streams for the two FlowEdit branches:
    # source history for source-prompt/source-latent calls, and edited history for
    # target-prompt/edited-latent calls. Using only one stream for both branches
    # either corrupts the source reference or collapses later chunks back to source.
    history_latents_src = torch.zeros(
        batch_size, C, num_history_latent_frames, H_lat, W_lat,
        dtype=torch.float32,
    )
    history_latents_tar = torch.zeros_like(history_latents_src)
    image_latents_src = None
    image_latents_tar = None
    output_frames = []

    # ── FlowEdit mode parameters ──────────────────────────────────────────────
    is_flowalign = (args.edit_type == "flowalign")

    # For FlowAlign (3-call), pass src_neg_embed=None so stage2_sample uses the 3-call path.
    # For FlowEdit (4-call), pass src_neg_embed for full CFG.
    flowedit_src_neg = None if is_flowalign else src_neg_embed

    pyramid_steps = args.pyramid_num_inference_steps_list
    total_steps = sum(pyramid_steps)

    # Decode edit_stage X = stage_idx + step_frac  (same convention as SDEdit).
    flowedit_start_stage = min(int(args.edit_stage), args.pyramid_num_stages - 1)
    step_frac = args.edit_stage - int(args.edit_stage)
    flowedit_start_step  = int(step_frac * pyramid_steps[flowedit_start_stage])

    # Offload VAE to CPU during denoising to reclaim ~0.5 GB VRAM
    vae.to("cpu")
    torch.cuda.empty_cache()

    # Number of transformer blocks to temporarily offload during per-chunk VAE decode.
    # Each block ≈ 1.5 GB; offloading 3 frees ~4.5 GB — enough for the VAE's ~2.4 GB peak.
    # PCIe transfer cost is ~0.3 s each way, far faster than CPU decode.
    _n_decode_offload = 3

    # stage2 is distilled — no pipeline-level CFG.
    # FlowEdit velocity scales are passed separately as flowedit_src_gs / guidance_scale.
    pipe._guidance_scale = 1.0

    edit_type_str = f"{args.edit_type}_X{args.edit_stage}"
    print(f"\nRunning {args.edit_type.upper()} (stage2 pyramid, edit_stage={args.edit_stage})...")
    print(f"  Entry: stage {flowedit_start_stage}, step {flowedit_start_step}"
          f"  |  Steps per stage: {pyramid_steps}  |  Total: {total_steps}")
    print(f"  Target gs={args.target_guidance_scale}" +
          (f"  Source gs={args.source_guidance_scale}" if not is_flowalign else f"  zeta={args.zeta_scale}"))

    with torch.no_grad():
        for k in range(num_chunks):
            is_first_chunk = k == 0

            # Build history slices
            lh_long_src, lh_mid_src, lh_1x_src = history_latents_src[:, :, -num_history_latent_frames:].split(history_sizes, dim=2)
            lh_long_tar, lh_mid_tar, lh_1x_tar = history_latents_tar[:, :, -num_history_latent_frames:].split(history_sizes, dim=2)
            if keep_first_frame:
                if image_latents_src is None and is_first_chunk:
                    lh_prefix_src = torch.zeros_like(lh_1x_src[:, :, :1])
                else:
                    lh_prefix_src = image_latents_src
                if image_latents_tar is None and is_first_chunk:
                    lh_prefix_tar = torch.zeros_like(lh_1x_tar[:, :, :1])
                else:
                    lh_prefix_tar = image_latents_tar
                lh_short_src = torch.cat([lh_prefix_src, lh_1x_src], dim=2)
                lh_short_tar = torch.cat([lh_prefix_tar, lh_1x_tar], dim=2)
            else:
                lh_long_src, lh_mid_src, lh_short_src = history_latents_src[:, :, -num_history_latent_frames:].split(history_sizes, dim=2)
                lh_long_tar, lh_mid_tar, lh_short_tar = history_latents_tar[:, :, -num_history_latent_frames:].split(history_sizes, dim=2)

            from tqdm import tqdm
            with tqdm(total=total_steps, desc=f"chunk {k+1}/{num_chunks}") as pbar:
                latents_out = pipe.stage2_sample(
                    # Pass X0_src as latents (full res); stage2_sample will downsample for FlowEdit init.
                    latents=X0_src[k].to(transformer_dtype).to(device),
                    pyramid_num_stages=args.pyramid_num_stages,
                    pyramid_num_inference_steps_list=pyramid_steps,
                    # Target prompt embeds (used for Vt_tar)
                    prompt_embeds=tar_pos_embed,
                    negative_prompt_embeds=tar_neg_embed,
                    guidance_scale=args.target_guidance_scale,
                    indices_hidden_states=indices_hidden_states,
                    indices_latents_history_short=indices_latents_history_short,
                    indices_latents_history_mid=indices_latents_history_mid,
                    indices_latents_history_long=indices_latents_history_long,
                    latents_history_short=lh_short_src.to(device=device, dtype=transformer_dtype),
                    latents_history_mid=lh_mid_src.to(device=device, dtype=transformer_dtype),
                    latents_history_long=lh_long_src.to(device=device, dtype=transformer_dtype),
                    latents_history_short_target=lh_short_tar.to(device=device, dtype=transformer_dtype),
                    latents_history_mid_target=lh_mid_tar.to(device=device, dtype=transformer_dtype),
                    latents_history_long_target=lh_long_tar.to(device=device, dtype=transformer_dtype),
                    device=device,
                    transformer_dtype=transformer_dtype,
                    generator=generator,
                    use_zero_init=False,
                    progress_bar=pbar,
                    # ── FlowEdit parameters ──────────────────────────────────
                    flowedit_X0_src=X0_src[k].to(transformer_dtype),
                    flowedit_src_pos_embeds=src_pos_embed,
                    flowedit_src_neg_embeds=flowedit_src_neg,  # None → FlowAlign
                    flowedit_src_gs=args.source_guidance_scale,
                    flowedit_start_stage=flowedit_start_stage,
                    flowedit_start_step=flowedit_start_step,
                    flowedit_zeta=args.zeta_scale if is_flowalign else 0.0,
                )

            latents_out = latents_out.to(torch.float32)
            latents_out_cpu = latents_out.cpu()
            history_latents_src = torch.cat([history_latents_src, X0_src[k].to(dtype=torch.float32)], dim=2)
            history_latents_tar = torch.cat([history_latents_tar, latents_out_cpu], dim=2)
            if keep_first_frame and is_first_chunk:
                image_latents_src = X0_src[k][:, :, 0:1].to(dtype=torch.float32)
                image_latents_tar = latents_out_cpu[:, :, 0:1].to(dtype=torch.float32)

            # Offload transformer blocks one-by-one to free GPU memory for VAE decode.
            # Strategy: load VAE first (so its footprint is counted), then check headroom.
            import gc as _gc
            del lh_long_src, lh_mid_src, lh_short_src, lh_long_tar, lh_mid_tar, lh_short_tar, latents_out_cpu
            _gc.collect()
            torch.cuda.empty_cache()
            _vae_decode_headroom = int(9.0 * 1024**3)  # 9 GB: feat_cache (~4 GB) + conv3d peak (4.75 GB)
            _n_actually_offloaded = 0
            vae.to(device=device, dtype=torch.bfloat16)
            for _i in range(len(transformer.blocks)):
                _gc.collect()
                torch.cuda.empty_cache()
                _free, _ = torch.cuda.mem_get_info()
                if _free >= _vae_decode_headroom:
                    break
                transformer.blocks[-(_i + 1)].to("cpu")
                _n_actually_offloaded += 1
            _gc.collect()
            torch.cuda.empty_cache()
            _free, _ = torch.cuda.mem_get_info()
            print(f"  [decode] offloaded {_n_actually_offloaded} blocks, free={_free/1024**3:.2f} GB")

            z_k = (latents_out.to(device=device, dtype=torch.float32) / latents_std + latents_mean).to(torch.bfloat16)
            del latents_out
            with torch.no_grad():
                decoded = vae.decode(z_k, return_dict=False)[0].float().cpu()
            del z_k
            output_frames.append(decoded)

            vae.to("cpu")
            for _i in range(_n_actually_offloaded - 1, -1, -1):
                transformer.blocks[-(_i + 1)].to(device)
            torch.cuda.empty_cache()

    output_video = torch.cat(output_frames, dim=2)
    output = pipe.video_processor.postprocess_video(output_video, output_type="np")

    file_count = len([f for f in os.listdir(args.output_folder) if os.path.isfile(os.path.join(args.output_folder, f))])
    output_path = os.path.join(
        args.output_folder,
        f"{file_count:04d}_{args.edit_type}_stage2_X{args.edit_stage}_{int(time.time())}.mp4",
    )
    export_to_video(output[0], output_path, fps=24)
    print(f"Saved: {output_path}")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
