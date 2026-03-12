# Example: VACE V2V inference with Helios-Distilled
# Requires a HeliosVACETransformer3DModel checkpoint at --transformer_path.
#
# Example: Running inference with 2-GPU parallelism
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 infer_helios_vace.py \
#     --enable_parallelism \

python infer_helios_vace.py \
    --base_model_path "BestWishYsh/Helios-Distilled" \
    --transformer_path "BestWishYsh/Helios-Distilled" \
    --vace_module_path "BestWishYsh/Wan2_1-VACE_module_14B_bf16.safetensors" \
    --source_video "example/car.mp4" \
    --prompt "A bright yellow Lamborghini Huracán Tecnica speeds along a curving mountain road, surrounded by lush green trees under a partly cloudy sky. The car's sleek design and vibrant color stand out against the natural backdrop, emphasizing its dynamic movement." \
    --num_frames 240 \
    --guidance_scale 1.0 \
    --is_enable_stage2 \
    --pyramid_num_inference_steps_list 2 2 2 \
    --is_amplify_first_chunk \
    --vace_context_scale 1.0 \
    --output_path "./output_helios/helios-vace/vace_output.mp4"
