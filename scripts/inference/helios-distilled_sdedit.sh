echo "=== Helios SDEdit (stage2 DMD) Test ==="

echo "car: Lambo -> Porsche"
python infer_sdedit_helios_stage2.py \
    --video_path "example/car.mp4" \
    --prompt "A silver Porsche 911 car speeds along a curving mountain road, surrounded by lush green trees under a partly cloudy sky." \
    --height 384 --width 640 --num_frames 33 \
    --guidance_scale 1.0 \
    --pyramid_num_stages 3 \
    --pyramid_num_inference_steps_list 4 4 4 \
    --edit_stage 1.0 \
    --output_folder "./output_helios/distilled_sdedit/lambo_porsche"