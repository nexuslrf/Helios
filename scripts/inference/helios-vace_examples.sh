#!/bin/bash
# ============================================================
# Helios VACE Inference Examples
#
# Exercises all VACE task types using the pre-processed assets
# from VACE-Benchmark:
#   /home/ruofan/Projects/VACE/benchmarks/VACE-Benchmark/assets/examples/
#
# Each task uses infer_helios_vace.py.
#
# Usage:
#   bash scripts/inference/helios-vace_examples.sh               # all tasks
#   bash scripts/inference/helios-vace_examples.sh depth         # one task
#   bash scripts/inference/helios-vace_examples.sh inpainting
# ============================================================

set -e
cd /home/ruofan/Projects/Helios
export PATH=~/miniconda3/envs/vwm/bin:$PATH

# ---- Shared paths ----
BASE_MODEL="BestWishYsh/Helios-Distilled"
VACE_MODULE="BestWishYsh/Wan2_1-VACE_module_14B_bf16.safetensors"
ASSETS=/home/ruofan/Projects/VACE/benchmarks/VACE-Benchmark/assets/examples
OUTDIR=output_helios/vace_examples

# ---- Generation defaults (distilled) ----
COMMON_ARGS="
  --base_model_path $BASE_MODEL
  --transformer_path $BASE_MODEL
  --vace_module_path $VACE_MODULE
  --num_frames 161
  --guidance_scale 1.0
  --is_enable_stage2
  --pyramid_num_inference_steps_list 2 2 2
  --is_amplify_first_chunk
  --vace_context_scale 1.0
  --weight_dtype bf16
"

mkdir -p "$OUTDIR"

# ---- Task selector ----
TARGET="${1:-all}"
run_example() {
    local name="$1"
    [[ "$TARGET" == "all" || "$TARGET" == "$name" ]] && return 0 || return 1
}

# ============================================================
# 1. v2v / depth  — depth-map control video
# ============================================================
if run_example "depth"; then
echo ""; echo "=== depth ==="
python infer_helios_vace.py $COMMON_ARGS \
    --source_video "$ASSETS/depth/src_video.mp4" \
    --prompt "一群年轻人在天空之城拍摄集体照。画面中，一对年轻情侣手牵手，轻声细语，相视而笑，周围是飞翔的彩色热气球和闪烁的星星，营造出浪漫的氛围。天空中，暖阳透过飘浮的云朵，洒下斑驳的光影。镜头以近景特写开始，随着情侣间的亲密互动，缓缓拉远。" \
    --output_path "$OUTDIR/depth.mp4"
fi

# ============================================================
# 2. v2v / flow  — optical-flow control video
# ============================================================
if run_example "flow"; then
echo ""; echo "=== flow ==="
python infer_helios_vace.py $COMMON_ARGS \
    --source_video "$ASSETS/flow/src_video.mp4" \
    --prompt "纪实摄影风格，一颗鲜红的小番茄缓缓落入盛着牛奶的玻璃杯中，溅起晶莹的水花。画面以慢镜头捕捉这一瞬间，水花在空中绽放，形成美丽的弧线。近景特写，垂直俯视视角。" \
    --output_path "$OUTDIR/flow.mp4"
fi

# ============================================================
# 3. v2v / gray  — grayscale (luminance) control video
# ============================================================
if run_example "gray"; then
echo ""; echo "=== gray ==="
python infer_helios_vace.py $COMMON_ARGS \
    --source_video "$ASSETS/gray/src_video.mp4" \
    --prompt "镜头缓缓向右平移，身穿淡黄色坎肩长裙的长发女孩面对镜头露出灿烂的漏齿微笑。她的长发随风轻扬，眼神明亮而充满活力。背景是秋天红色和黄色的树叶。中景人像，强调自然光效。" \
    --output_path "$OUTDIR/gray.mp4"
fi

# ============================================================
# 4. v2v / pose  — pose-skeleton control video
# ============================================================
if run_example "pose"; then
echo ""; echo "=== pose ==="
python infer_helios_vace.py $COMMON_ARGS \
    --source_video "$ASSETS/pose/src_video.mp4" \
    --prompt "在一个热带的庆祝派对上，一家人围坐在椰子树下的长桌旁。桌上摆满了异国风味的美食。长辈们愉悦地交谈，年轻人兴奋地举杯碰撞，孩子们在沙滩上欢乐奔跑。" \
    --output_path "$OUTDIR/pose.mp4"
fi

# ============================================================
# 5. v2v / scribble  — edge/scribble control video
# ============================================================
if run_example "scribble"; then
echo ""; echo "=== scribble ==="
python infer_helios_vace.py $COMMON_ARGS \
    --source_video "$ASSETS/scribble/src_video.mp4" \
    --prompt "画面中荧光色彩的无人机从极低空高速掠过超现实主义风格的西安古城墙，尘埃反射着阳光。整体画质清晰华丽，运镜流畅如水。" \
    --output_path "$OUTDIR/scribble.mp4"
fi

# ============================================================
# 6. v2v / layout  — bounding-box trajectory control video
# ============================================================
if run_example "layout"; then
echo ""; echo "=== layout ==="
python infer_helios_vace.py $COMMON_ARGS \
    --source_video "$ASSETS/layout/src_video.mp4" \
    --prompt "视频展示了一只成鸟在树枝上的巢中喂养它的幼鸟。成鸟在喂食的过程中，幼鸟张开嘴巴等待食物。随后，成鸟飞走，幼鸟继续等待。背景是模糊的绿色植被。" \
    --output_path "$OUTDIR/layout.mp4"
fi

# ============================================================
# 7. inpainting  — masked region regeneration
#    src_mask: white = regenerate, black = preserve
# ============================================================
if run_example "inpainting"; then
echo ""; echo "=== inpainting ==="
python infer_helios_vace.py $COMMON_ARGS \
    --source_video "$ASSETS/inpainting/src_video.mp4" \
    --mask_path    "$ASSETS/inpainting/src_mask.mp4" \
    --prompt "一只巨大的金色凤凰从繁华的城市上空展翅飞过，羽毛如火焰般璀璨，闪烁着温暖的光辉，翅膀雄伟地展开。下方是熙熙攘攘的市中心，人群惊叹，车水马龙。" \
    --output_path "$OUTDIR/inpainting.mp4"
fi

# ============================================================
# 8. outpainting  — border region expansion
# ============================================================
if run_example "outpainting"; then
echo ""; echo "=== outpainting ==="
python infer_helios_vace.py $COMMON_ARGS \
    --source_video "$ASSETS/outpainting/src_video.mp4" \
    --mask_path    "$ASSETS/outpainting/src_mask.mp4" \
    --prompt "赛博朋克风格，无人机俯瞰视角下的现代西安城墙，镜头穿过永宁门时泛起金色涟漪，城墙砖块化作数据流重组为唐代长安城。周围的街道上流动的人群和飞驰的机械交通工具交织在一起。" \
    --output_path "$OUTDIR/outpainting.mp4"
fi

# ============================================================
# 9. firstframe  — extend video from a single reference frame
#    src_mask: first frame = black (preserve), rest = white (generate)
# ============================================================
if run_example "firstframe"; then
echo ""; echo "=== firstframe ==="
python infer_helios_vace.py $COMMON_ARGS \
    --source_video "$ASSETS/firstframe/src_video.mp4" \
    --mask_path    "$ASSETS/firstframe/src_mask.mp4" \
    --prompt "纪实摄影风格，前景是一位中国越野爱好者坐在越野车上，手持车载电台正在进行通联。他五官清晰，表情专注，眼神坚定地望向前方。镜头从车外缓缓拉近，最后定格在人物的面部特写。" \
    --output_path "$OUTDIR/firstframe.mp4"
fi

# ============================================================
# 10. swap_anything  — replace masked region with new content
# ============================================================
if run_example "swap_anything"; then
echo ""; echo "=== swap_anything ==="
python infer_helios_vace.py $COMMON_ARGS \
    --source_video "$ASSETS/swap_anything/src_video.mp4" \
    --mask_path    "$ASSETS/swap_anything/src_mask.mp4" \
    --prompt "视频展示了一个人在宽阔的草原上骑马。他有淡紫色长发，穿着传统服饰白上衣黑裤子，动画建模画风。背景是壮观的山脉和多云的天空，给人一种宁静而广阔的感觉。" \
    --output_path "$OUTDIR/swap_anything.mp4"
fi

echo ""
echo "============================================================"
echo "  All examples complete. Outputs saved to: $OUTDIR"
echo "============================================================"
