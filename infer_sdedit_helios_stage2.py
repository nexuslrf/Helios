"""
SDEdit inference script for Helios using the stage2 DMD pyramid pipeline.

Optionally accepts a reference image (--image_path) to enable I2V-style conditioning:
  - The reference image seeds the lh_short prefix slot (noised with image_noise_sigma).
  - fake_image_latents (last frame of a static-video encode) seeds the most-recent
    history slot (noised with video_noise_sigma), matching pipeline_helios_diffusers.py.
  - --ref_frame_strength blends the first-frame start latent toward the reference image
    trajectory, anchoring the appearance of frame 0.

Edit strength is expressed as `edit_stage` (float X) instead of a raw noise level:

  - X = 0.0  → start at the very beginning of stage 0 (maximum edit, pure noise)
  - X = 1.5  → start at step int(0.5 * T1) of stage 1 (partial edit, coarse source preserved)
  - X = 2.0  → start at the beginning of stage 2 (minimum edit, fine details from source)

The noise injection follows the training pyramid structure (utils_helios_base.py:868):

  Stage 0 start_point  = noise  (pure noise at coarsest resolution)
  Stage k start_point  = start_sigma[k] * noise_k + (1 - start_sigma[k]) * upsample(x0_src_{k-1})

This is the same formula used in prepare_stage2_clean_input, so the noisy input x_t lies
exactly on the training distribution at every stage and timestep.

The DMD noise anchor for stage2_sample is start_point (not x_t itself).
"""

import gc
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

from helios.diffusers_version.pipeline_helios_sdedit import HeliosPipelineSDEdit
from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler
from helios.diffusers_version.transformer_helios_diffusers import HeliosTransformer3DModel
from helios.modules.helios_kernels import (
    replace_all_norms_with_flash_norms,
    replace_rmsnorm_with_fp32,
    replace_rope_with_flash_rope,
)

from diffusers.models import AutoencoderKLWan
from diffusers.utils import export_to_video, load_image, load_video
from diffusers.utils.torch_utils import randn_tensor


def parse_args():
    parser = argparse.ArgumentParser(description="SDEdit (stage2 pyramid DMD) video editing with Helios")

    parser.add_argument("--base_model_path", type=str, default="BestWishYsh/Helios-Distilled")
    parser.add_argument("--transformer_path", type=str, default="BestWishYsh/Helios-Distilled")
    parser.add_argument("--output_folder", type=str, default="./output_helios/sdedit_stage2")

    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument(
        "--image_path", type=str, default=None,
        help="Optional reference image for I2V-style appearance conditioning.",
    )
    parser.add_argument(
        "--edit_stage",
        type=float,
        default=0.5,
        help=(
            "Edit position in pyramid schedule (float X). "
            "Integer part = which pyramid stage (0, 1, 2). "
            "Fractional part = how far into that stage (0=beginning, 0.9=near end). "
            "Lower X = more edit; higher X = more source preserved. "
            "E.g. 0.5 = mid-stage 0, 1.0 = start of stage 1, 1.5 = mid-stage 1."
        ),
    )

    parser.add_argument("--prompt", type=str, required=True)
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
    parser.add_argument("--guidance_scale", type=float, default=5.0)
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

    # I2V args (only active when --image_path is provided)
    parser.add_argument(
        "--image_noise_sigma_min", type=float, default=0.111,
        help="Min noise sigma for the reference image prefix slot in lh_short.",
    )
    parser.add_argument(
        "--image_noise_sigma_max", type=float, default=0.135,
        help="Max noise sigma for the reference image prefix slot in lh_short.",
    )
    parser.add_argument(
        "--video_noise_sigma_min", type=float, default=0.111,
        help="Min noise sigma for fake_image_latents injected into history last slot.",
    )
    parser.add_argument(
        "--video_noise_sigma_max", type=float, default=0.135,
        help="Max noise sigma for fake_image_latents injected into history last slot.",
    )
    parser.add_argument(
        "--ref_frame_strength", type=float, default=1.0,
        help=(
            "How strongly to replace the first-frame start latent with the reference image "
            "trajectory. 1.0 = full replacement, 0.0 = source video only."
        ),
    )

    return parser.parse_args()


def build_latent_pyramid(latents, pyramid_num_stages):
    """Downsample clean latents to each pyramid stage resolution.

    Returns list indexed [0=coarsest, ..., pyramid_num_stages-1=full].
    Matches the pyramid_latent_list in prepare_stage2_clean_input.
    """
    pyramid = [latents]
    batch_size, num_channel, num_frames, height, width = latents.shape
    h, w = height, width
    cur = latents
    for _ in range(pyramid_num_stages - 1):
        h //= 2
        w //= 2
        cur_2d = cur.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, num_channel, cur.shape[-2], cur.shape[-1])
        cur_2d = F.interpolate(cur_2d, size=(h, w), mode="bilinear")
        cur = cur_2d.reshape(batch_size, num_frames, num_channel, h, w).permute(0, 2, 1, 3, 4)
        pyramid.append(cur)
    # pyramid[0] = full res, pyramid[-1] = coarsest → reverse to [coarsest, ..., full]
    return list(reversed(pyramid))


def build_noise_pyramid(noise, pyramid_num_stages):
    """Downsample noise to each pyramid stage resolution (bilinear + ×2 scaling).

    Returns list indexed [0=coarsest, ..., pyramid_num_stages-1=full].
    Matches noise_list in prepare_stage2_clean_input.
    """
    batch_size, num_channel, num_frames, height, width = noise.shape
    pyramid = [noise]
    h, w = height, width
    cur = noise
    for _ in range(pyramid_num_stages - 1):
        h //= 2
        w //= 2
        cur_2d = cur.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, num_channel, cur.shape[-2], cur.shape[-1])
        cur_2d = F.interpolate(cur_2d, size=(h, w), mode="bilinear") * 2
        cur = cur_2d.reshape(batch_size, num_frames, num_channel, h, w).permute(0, 2, 1, 3, 4)
        pyramid.append(cur)
    return list(reversed(pyramid))


def compute_sdedit_input(scheduler, latent_pyramid, noise_pyramid, pyramid_num_stages, stage_idx, step_idx, T_stage):
    """Compute the SDEdit noisy input x_t and DMD anchor start_point.

    Mirrors the training formula in prepare_stage2_clean_input:

      stage 0:  start_point = noise_pyramid[0]
      stage k:  start_point = start_sigma[k]*noise_k + (1-start_sigma[k])*upsample(x0_k-1)

      end_point (not last stage) = end_sigma[k]*noise_k + (1-end_sigma[k])*x0_k
      end_point (last stage)     = x0_k

      x_t = sigma_t * start_point + (1 - sigma_t) * end_point
    """
    dtype = latent_pyramid[stage_idx].dtype

    # start_point for this stage
    if stage_idx == 0:
        start_point = noise_pyramid[0]
    else:
        # upsample coarser clean latent to current stage resolution
        prev_clean = latent_pyramid[stage_idx - 1]
        batch_size, num_channel, num_frames = prev_clean.shape[:3]
        tgt_h, tgt_w = latent_pyramid[stage_idx].shape[-2], latent_pyramid[stage_idx].shape[-1]
        prev_2d = prev_clean.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, num_channel, prev_clean.shape[-2], prev_clean.shape[-1])
        prev_2d = F.interpolate(prev_2d, size=(tgt_h, tgt_w), mode="nearest")
        prev_clean_up = prev_2d.reshape(batch_size, num_frames, num_channel, tgt_h, tgt_w).permute(0, 2, 1, 3, 4)
        start_sigma_k = scheduler.start_sigmas[stage_idx]
        start_point = start_sigma_k * noise_pyramid[stage_idx] + (1 - start_sigma_k) * prev_clean_up

    # end_point for this stage
    if stage_idx == pyramid_num_stages - 1:
        end_point = latent_pyramid[stage_idx]
    else:
        end_sigma_k = scheduler.end_sigmas[stage_idx]
        end_point = end_sigma_k * noise_pyramid[stage_idx] + (1 - end_sigma_k) * latent_pyramid[stage_idx]

    # sigma at step_idx in the full T-step schedule for this stage
    s0 = float(scheduler.sigmas_per_stage[stage_idx][0].item())   # ≈ 0.999
    s1 = float(scheduler.sigmas_per_stage[stage_idx][-1].item())  # ≈ ~0
    if T_stage > 1:
        sigma_t = s0 + (s1 - s0) * step_idx / (T_stage - 1)
    else:
        sigma_t = s0

    # x_t: flow-matched noisy input
    x_t = sigma_t * start_point + (1 - sigma_t) * end_point

    return x_t.to(dtype), start_point.to(dtype)


def main():
    args = parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    weight_dtype = dtype_map[args.weight_dtype]

    os.makedirs(args.output_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # ---- Load models ----
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
    pipe = HeliosPipelineSDEdit.from_pretrained(
        args.base_model_path, transformer=transformer, vae=vae, scheduler=scheduler,
        torch_dtype=weight_dtype,
    ).to(device)

    transformer_dtype = transformer.dtype

    # ---- Parse edit_stage ----
    X = args.edit_stage
    stage_idx = int(X)
    step_frac = X % 1.0
    pyramid_num_stages = args.pyramid_num_stages
    pyramid_steps = args.pyramid_num_inference_steps_list

    assert stage_idx < pyramid_num_stages, (
        f"edit_stage {X} implies stage_idx={stage_idx} but pyramid_num_stages={pyramid_num_stages}"
    )
    T_stage = pyramid_steps[stage_idx]
    step_idx = int(step_frac * T_stage)  # 0 = beginning of stage, T-1 = near end

    print(f"edit_stage={X} → stage {stage_idx}, step {step_idx}/{T_stage}")

    # ---- Encode prompts ----
    print("Encoding prompts...")
    pos_embed, neg_embed = pipe.encode_prompt(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=device,
    )
    pos_embed = pos_embed.to(transformer_dtype)
    neg_embed = neg_embed.to(transformer_dtype)

    # ---- Shared VAE statistics ----
    latents_mean = (
        torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, vae.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, vae.dtype)

    height, width = args.height, args.width
    num_latent_frames_per_chunk = args.num_latent_frames_per_chunk
    vae_temporal = pipe.vae_scale_factor_temporal   # 4
    vae_spatial = pipe.vae_scale_factor_spatial     # 8
    min_frames = (num_latent_frames_per_chunk - 1) * vae_temporal + 1  # 33

    # ---- Encode reference image (I2V mode) ----
    image_latents = None    # prefix for lh_short; None → zeros prefix (pure SDEdit)
    fake_image_latents = None  # last-slot history seed; None → zeros history
    ref_latents = None      # clean single-frame ref (used for ref_frame_strength)

    if args.image_path is not None:
        print(f"Encoding reference image: {args.image_path}")
        ref_image_pil = load_image(args.image_path).resize((width, height))

        # Single-frame encode → image prefix (matches pipeline prepare_image_latents)
        ref_img_tensor = pipe.video_processor.preprocess_video(
            [ref_image_pil], height=height, width=width
        ).to(device=device, dtype=vae.dtype)
        with torch.no_grad():
            ref_latents = vae.encode(ref_img_tensor).latent_dist.sample(generator=generator)
            ref_latents = (ref_latents - latents_mean) * latents_std  # [1, C, 1, H_lat, W_lat]
            ref_latents = ref_latents.to(torch.float32)
        del ref_img_tensor

        sigma = (
            torch.rand(1, device=device, generator=generator)
            * (args.image_noise_sigma_max - args.image_noise_sigma_min)
            + args.image_noise_sigma_min
        )
        ref_noised = (1 - sigma) * ref_latents + sigma * torch.randn_like(ref_latents)
        image_latents = ref_noised
        print(f"  ref image encoded; latent shape={ref_latents.shape}, sigma={sigma.item():.4f}")

        # fake_image_latents: last frame of full static-video encode (3D-conv aware).
        # Matches pipeline prepare_image_latents fake_latents branch.
        print("  encoding fake_image_latents (static video last frame)...")
        ref_fake_tensor = pipe.video_processor.preprocess_video(
            [ref_image_pil] * min_frames, height=height, width=width
        ).to(device=device, dtype=vae.dtype)
        with torch.no_grad():
            fake_latents_full = vae.encode(ref_fake_tensor).latent_dist.sample(generator=generator)
            fake_latents_full = (fake_latents_full - latents_mean) * latents_std
            fake_image_latents = fake_latents_full[:, :, -1:, :, :].to(torch.float32)
        del ref_fake_tensor, fake_latents_full
        fake_noise_sigma = (
            torch.rand(1, device=device, generator=generator)
            * (args.video_noise_sigma_max - args.video_noise_sigma_min)
            + args.video_noise_sigma_min
        )
        fake_image_latents = (
            (1 - fake_noise_sigma) * fake_image_latents
            + fake_noise_sigma * torch.randn_like(fake_image_latents)
        )
        print(f"  fake_image_latents encoded; shape={fake_image_latents.shape}, sigma={fake_noise_sigma.item():.4f}")

    # ---- Load and encode source video ----
    print(f"Loading source video: {args.video_path}")
    source_video = load_video(args.video_path)

    video_tensor = pipe.video_processor.preprocess_video(source_video, height=height, width=width)
    video_tensor = video_tensor.to(device=device, dtype=vae.dtype)

    num_video_frames = video_tensor.shape[2]
    num_chunks = max(1, num_video_frames // min_frames)
    total_valid_frames = num_chunks * min_frames
    start_frame = num_video_frames - total_valid_frames

    print(f"Video: {num_video_frames} frames → {num_chunks} chunks of {min_frames} frames each")

    print("Encoding source video to latents...")
    X0_src = []
    with torch.no_grad():
        for k in range(num_chunks):
            chunk_start = start_frame + k * min_frames
            chunk_end = chunk_start + min_frames
            chunk = video_tensor[:, :, chunk_start:chunk_end]
            z = vae.encode(chunk).latent_dist.sample(generator=generator)
            z = (z - latents_mean) * latents_std
            X0_src.append(z.to(torch.float32))
    del video_tensor

    # ---- Build pyramid noisy inputs per chunk ----
    # Each chunk gets an independent noise draw; x_t is computed following the
    # training formula (prepare_stage2_clean_input) at the chosen edit_stage.
    start_latents_list = []  # x_t at stage_idx resolution
    noise_anchors_list = []  # start_point for DMD anchor
    eps_chunk0 = None        # saved for ref_frame_strength (I2V only)

    with torch.no_grad():
        for k in range(num_chunks):
            x0_k = X0_src[k].to(device=device, dtype=torch.float32)
            eps_k = randn_tensor(x0_k.shape, generator=generator, device=device, dtype=torch.float32)
            if k == 0 and ref_latents is not None and args.ref_frame_strength > 0.0:
                eps_chunk0 = eps_k

            lat_pyr = build_latent_pyramid(x0_k, pyramid_num_stages)
            noise_pyr = build_noise_pyramid(eps_k, pyramid_num_stages)

            x_t_k, sp_k = compute_sdedit_input(
                pipe.scheduler, lat_pyr, noise_pyr, pyramid_num_stages, stage_idx, step_idx, T_stage
            )
            start_latents_list.append(x_t_k.to(transformer_dtype))
            noise_anchors_list.append(sp_k.to(transformer_dtype))

    # ---- First-frame start latent: replace with reference image trajectory (I2V) ----
    if eps_chunk0 is not None:
        with torch.no_grad():
            ref_lat_pyr = build_latent_pyramid(ref_latents.to(device=device), pyramid_num_stages)
            eps_ref = eps_chunk0[:, :, 0:1].to(device=device)
            noise_pyr_ref = build_noise_pyramid(eps_ref, pyramid_num_stages)
            x_t_ref, _ = compute_sdedit_input(
                pipe.scheduler, ref_lat_pyr, noise_pyr_ref, pyramid_num_stages, stage_idx, step_idx, T_stage
            )
            s = args.ref_frame_strength
            first_frame_blended = (
                s * x_t_ref.to(transformer_dtype)
                + (1 - s) * start_latents_list[0][:, :, 0:1]
            )
            start_latents_list[0] = torch.cat([first_frame_blended, start_latents_list[0][:, :, 1:]], dim=2)
        print(f"  ref_frame_strength={s}: first frame start latent → reference image trajectory")
        del eps_chunk0

    if ref_latents is not None:
        del ref_latents

    # ---- Prepare positional indices ----
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

    # History buffer: zeros init; in I2V mode seed the most-recent slot with
    # fake_image_latents (mirrors pipeline_helios_diffusers.py L1168-1170).
    history_latents = torch.zeros(
        batch_size, C, num_history_latent_frames, H_lat, W_lat,
        device=device, dtype=torch.float32,
    )
    if fake_image_latents is not None:
        history_latents[:, :, -1:, :, :] = fake_image_latents.to(device=device)
        del fake_image_latents

    # Set guidance scale on pipe (used by do_classifier_free_guidance property)
    pipe._guidance_scale = args.guidance_scale

    # Compute total steps (stages before stage_idx are skipped, entry stage may be truncated)
    steps_entry = T_stage - step_idx
    steps_after = sum(pyramid_steps[stage_idx + 1:])
    total_steps = steps_entry + steps_after

    print(f"\nRunning SDEdit{'+ I2V' if args.image_path else ''} (stage2 pyramid, edit_stage={X})...")
    output_frames = []

    with torch.no_grad():
        for k in range(num_chunks):
            is_first_chunk = k == 0

            # Build history slices
            lh_long, lh_mid, lh_1x = history_latents[:, :, -num_history_latent_frames:].split(history_sizes, dim=2)
            if keep_first_frame:
                if image_latents is None and is_first_chunk:
                    lh_prefix = torch.zeros_like(lh_1x[:, :, :1])
                else:
                    lh_prefix = image_latents
                lh_short = torch.cat([lh_prefix, lh_1x], dim=2)
            else:
                lh_long, lh_mid, lh_short = history_latents[:, :, -num_history_latent_frames:].split(history_sizes, dim=2)

            from tqdm import tqdm
            with tqdm(total=total_steps, desc=f"chunk {k+1}/{num_chunks}") as pbar:
                latents_out = pipe.stage2_sample(
                    latents=start_latents_list[k],
                    pyramid_num_stages=pyramid_num_stages,
                    pyramid_num_inference_steps_list=pyramid_steps,
                    prompt_embeds=pos_embed,
                    negative_prompt_embeds=neg_embed,
                    guidance_scale=args.guidance_scale,
                    indices_hidden_states=indices_hidden_states,
                    indices_latents_history_short=indices_latents_history_short,
                    indices_latents_history_mid=indices_latents_history_mid,
                    indices_latents_history_long=indices_latents_history_long,
                    latents_history_short=lh_short.to(transformer_dtype),
                    latents_history_mid=lh_mid.to(transformer_dtype),
                    latents_history_long=lh_long.to(transformer_dtype),
                    device=device,
                    transformer_dtype=transformer_dtype,
                    generator=generator,
                    sdedit_start_stage=stage_idx,
                    sdedit_start_step=step_idx,
                    sdedit_noise_anchor=noise_anchors_list[k],
                    use_zero_init=False,
                    progress_bar=pbar,
                )

            latents_out = latents_out.to(torch.float32)
            history_latents = torch.cat([history_latents, latents_out], dim=2)
            if keep_first_frame and is_first_chunk:
                image_latents = latents_out[:, :, 0:1]

            # ---- GPU decode per chunk (dynamic transformer block offload) ----
            _vae_decode_headroom = int(9.0 * 1024**3)
            _n_offloaded = 0
            vae.to(device=device, dtype=torch.bfloat16)
            for _i in range(len(transformer.blocks)):
                gc.collect()
                torch.cuda.empty_cache()
                _free, _ = torch.cuda.mem_get_info()
                if _free >= _vae_decode_headroom:
                    break
                transformer.blocks[-(_i + 1)].to("cpu")
                _n_offloaded += 1
            gc.collect()
            torch.cuda.empty_cache()
            _free, _ = torch.cuda.mem_get_info()
            print(f"  [decode] offloaded {_n_offloaded} blocks, free={_free/1024**3:.2f} GB")

            z_k = (latents_out.to(device=device, dtype=torch.float32) / latents_std + latents_mean).to(torch.bfloat16)
            del latents_out
            decoded = vae.decode(z_k, return_dict=False)[0].float().cpu()
            del z_k
            output_frames.append(decoded)

            vae.to("cpu")
            for _i in range(_n_offloaded - 1, -1, -1):
                transformer.blocks[-(_i + 1)].to(device)
            torch.cuda.empty_cache()

    output_video = torch.cat(output_frames, dim=2)
    output = pipe.video_processor.postprocess_video(output_video, output_type="np")

    file_count = len([f for f in os.listdir(args.output_folder) if os.path.isfile(os.path.join(args.output_folder, f))])
    suffix = "_i2v" if args.image_path else ""
    output_path = os.path.join(
        args.output_folder,
        f"{file_count:04d}_sdedit_stage2{suffix}_x{args.edit_stage:.2f}_{int(time.time())}.mp4",
    )
    export_to_video(output[0], output_path, fps=24)
    print(f"Saved: {output_path}")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
