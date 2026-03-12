"""
Video-to-Video inference using VACE conditioning on Helios.

Source video is encoded into 96-channel VACE conditioning latents
(inactive + reactive + tiled mask). Denoising starts from pure noise,
guided by per-chunk VACE hints derived from the source video.

vace_context_scale controls how strongly the source structure influences
the output (0.0 = no conditioning, 1.0 = normal, 2.0 = strong).

Example (distilled, single GPU):
  python infer_helios_vace.py \\
      --base_model_path BestWishYsh/Helios-Distilled \\
      --transformer_path /path/to/helios-vace-transformer \\
      --source_video example/car.mp4 \\
      --prompt "A bright yellow Lamborghini speeds along a mountain road." \\
      --output_path ./output_helios/vace_output.mp4 \\
      --num_frames 240 \\
      --guidance_scale 1.0 \\
      --is_enable_stage2 \\
      --pyramid_num_inference_steps_list 2 2 2 \\
      --is_amplify_first_chunk \\
      --vace_context_scale 1.0
"""

import importlib
import os
import time
import argparse

import torch
import torch.nn.functional as F

if importlib.util.find_spec("torch_npu") is not None:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

from helios.diffusers_version.pipeline_helios_vace_diffusers import HeliosVACEPipeline
from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler
from helios.diffusers_version.transformer_helios_vace_diffusers import HeliosVACETransformer3DModel
from helios.diffusers_version.transformer_helios_diffusers import HeliosTransformer3DModel
from helios.modules.helios_kernels import (
    replace_all_norms_with_flash_norms,
    replace_rmsnorm_with_fp32,
    replace_rope_with_flash_rope,
)

from diffusers.models import AutoencoderKLWan
from diffusers.utils import export_to_video


def parse_args():
    parser = argparse.ArgumentParser(description="VACE V2V inference with Helios")

    # === Model paths ===
    parser.add_argument("--base_model_path", type=str, default="BestWishYsh/Helios-Distilled",
                        help="HF repo or local path for VAE, scheduler, text encoder, etc.")
    parser.add_argument("--transformer_path", type=str, default=None,
                        help="Path to base HeliosTransformer3DModel weights. "
                             "Defaults to --base_model_path if not set.")
    parser.add_argument("--vace_module_path", type=str, default=None,
                        help="Path to a Wan2.1-VACE-module safetensors file that provides "
                             "vace_patch_embedding and vace_blocks weights "
                             "(e.g. BestWishYsh/Wan2_1-VACE_module_14B_bf16.safetensors). "
                             "These are loaded on top of the base transformer weights.")
    parser.add_argument("--output_path", type=str, default="./output_helios/vace_output.mp4")
    parser.add_argument("--enable_compile", action="store_true")
    parser.add_argument("--low_vram_mode", action="store_true")

    # === Source video / mask ===
    parser.add_argument("--source_video", type=str, required=True,
                        help="Path to the source video for VACE conditioning.")
    parser.add_argument("--mask_path", type=str, default=None,
                        help="Optional path to a single-channel mask video (grayscale). "
                             "If not provided, an all-ones mask (full-frame control) is used.")

    # === Generation parameters ===
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, "
                "paintings, images, static, overall gray, worst quality, low quality, "
                "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
                "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
                "still picture, messy background, three legs, many people in the background, "
                "walking backwards",
    )
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=99,
                        help="Number of pixel frames to generate.")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])

    # === VACE ===
    parser.add_argument("--vace_context_scale", type=float, default=1.0,
                        help="Scale factor for VACE hint injection "
                             "(0.0 = no conditioning, 1.0 = normal, 2.0 = strong).")

    # === AR / pyramid ===
    parser.add_argument("--num_latent_frames_per_chunk", type=int, default=9)
    parser.add_argument("--is_enable_stage2", action="store_true")
    parser.add_argument("--pyramid_num_inference_steps_list", type=int, nargs="+",
                        default=[20, 20, 20])
    parser.add_argument("--is_amplify_first_chunk", action="store_true")
    parser.add_argument("--is_skip_first_chunk", action="store_true")
    parser.add_argument("--use_zero_init", action="store_true")
    parser.add_argument("--zero_steps", type=int, default=1)

    return parser.parse_args()


def load_video_tensor(path: str, height: int, width: int, num_frames: int) -> torch.Tensor:
    """Load a video file and return a float tensor of shape [1, C, T, H, W] in [-1, 1]."""
    from torchvision.io import read_video

    video, _, _ = read_video(path, pts_unit="sec", output_format="TCHW")
    video = video.float() / 255.0 * 2.0 - 1.0  # [T, C, H, W] in [-1, 1]
    video = video.permute(1, 0, 2, 3)           # [C, T, H, W]

    # Resize spatial dimensions
    C, T, H, W = video.shape
    if (H, W) != (height, width):
        video = F.interpolate(
            video.permute(1, 0, 2, 3),   # [T, C, H, W]
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).permute(1, 0, 2, 3)            # [C, T, H, W]

    # Trim or pad to target frame count
    if video.shape[1] > num_frames:
        video = video[:, :num_frames]
    elif video.shape[1] < num_frames:
        pad = video[:, -1:].expand(-1, num_frames - video.shape[1], -1, -1)
        video = torch.cat([video, pad], dim=1)

    return video.unsqueeze(0)  # [1, C, T, H, W]


def main():
    args = parse_args()
    print(f"Args: {args}")
    assert not (args.low_vram_mode and args.enable_compile), (
        "--low_vram_mode and --enable_compile cannot be used together."
    )

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    weight_dtype = dtype_map[args.weight_dtype]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load model components
    # ------------------------------------------------------------------ #
    transformer_path = args.transformer_path or args.base_model_path

    # Step 1: Load the base Helios transformer normally.
    #         from_pretrained with init_empty_weights leaves any keys not in the
    #         checkpoint on the meta device.  Loading the base model first avoids
    #         this: all its tensors are on CPU with real data.
    print(f"Loading base transformer from {transformer_path}")
    base_transformer = HeliosTransformer3DModel.from_pretrained(
        transformer_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    )

    # Step 2: Construct HeliosVACETransformer3DModel with the same config.
    #         All parameters (including vace_patch_embedding / vace_blocks) are
    #         allocated on CPU with real (random) data — no meta tensors.
    #         Filter out diffusers internal metadata keys (prefixed with "_").
    print("Building HeliosVACETransformer3DModel …")
    base_init_kwargs = {k: v for k, v in base_transformer.config.items() if not k.startswith("_")}
    transformer = HeliosVACETransformer3DModel(
        **base_init_kwargs,
        vace_layers=[0, 5, 10, 15, 20, 25, 30, 35],
        vace_in_channels=96,
    ).to(dtype=weight_dtype)

    # Step 3: Copy base transformer weights into the VACE model.
    #         VACE-specific keys (vace_patch_embedding, vace_blocks) are absent
    #         from the base state dict and remain randomly initialised here.
    missing, unexpected = transformer.load_state_dict(
        base_transformer.state_dict(), strict=False
    )
    vace_missing = [k for k in missing if k.startswith(("vace_patch_embedding", "vace_blocks"))]
    non_vace_missing = [k for k in missing if k not in vace_missing]
    if non_vace_missing:
        print(f"WARNING: non-VACE keys missing after base weight copy: {non_vace_missing}")
    del base_transformer  # free memory

    # Step 4: Load VACE module weights on top.
    if args.vace_module_path is not None:
        print(f"Loading VACE module weights from {args.vace_module_path}")
        transformer.load_vace_module(args.vace_module_path)
    else:
        print("WARNING: --vace_module_path not set; VACE blocks are randomly initialised.")

    if not args.enable_compile:
        transformer = replace_rmsnorm_with_fp32(transformer)
        transformer = replace_all_norms_with_flash_norms(transformer)
        replace_rope_with_flash_rope()
    try:
        transformer.set_attention_backend("_flash_3_hub")
    except Exception:
        transformer.set_attention_backend("flash_hub")

    vae = AutoencoderKLWan.from_pretrained(
        args.base_model_path, subfolder="vae", torch_dtype=torch.float32
    )
    scheduler = HeliosScheduler.from_pretrained(
        args.base_model_path, subfolder="scheduler"
    )
    pipe = HeliosVACEPipeline.from_pretrained(
        args.base_model_path,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=weight_dtype,
    )

    if args.enable_compile:
        torch.backends.cudnn.benchmark = True
        pipe.text_encoder.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        pipe.vae.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        pipe.transformer.compile(mode="max-autotune-no-cudagraphs", dynamic=False)

    if args.low_vram_mode:
        pipe.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=1,
            use_stream=True,
            record_stream=True,
        )
    else:
        pipe = pipe.to(device)

    # ------------------------------------------------------------------ #
    # Prepare source video and mask
    # ------------------------------------------------------------------ #
    print(f"Loading source video: {args.source_video}")
    control_video = load_video_tensor(
        args.source_video, args.height, args.width, args.num_frames
    ).to(device=device, dtype=weight_dtype)
    # control_video: [1, C, T, H, W] in [-1, 1]

    control_mask = None
    if args.mask_path is not None:
        print(f"Loading mask from: {args.mask_path}")
        mask_raw = load_video_tensor(args.mask_path, args.height, args.width, args.num_frames)
        # Collapse to single channel if RGB
        control_mask = mask_raw.mean(dim=1, keepdim=True).to(device=device, dtype=weight_dtype)
    # If control_mask is None, HeliosVACEPipeline will use an all-ones mask.

    # ------------------------------------------------------------------ #
    # Run generation
    # ------------------------------------------------------------------ #
    print(f"Running VACE V2V generation (vace_context_scale={args.vace_context_scale})...")
    with torch.no_grad():
        output = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            # AR / pyramid
            history_sizes=[16, 2, 1],
            num_latent_frames_per_chunk=args.num_latent_frames_per_chunk,
            keep_first_frame=True,
            is_enable_stage2=args.is_enable_stage2,
            pyramid_num_inference_steps_list=args.pyramid_num_inference_steps_list,
            is_skip_first_chunk=args.is_skip_first_chunk,
            is_amplify_first_chunk=args.is_amplify_first_chunk,
            use_zero_init=args.use_zero_init,
            zero_steps=args.zero_steps,
            # VACE
            control_video=control_video,
            control_mask=control_mask,
            conditioning_scale=args.vace_context_scale,
        ).frames[0]

    # ------------------------------------------------------------------ #
    # Save output
    # ------------------------------------------------------------------ #
    export_to_video(output, args.output_path, fps=args.fps)
    print(f"Saved output video to: {args.output_path}")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")


if __name__ == "__main__":
    main()
