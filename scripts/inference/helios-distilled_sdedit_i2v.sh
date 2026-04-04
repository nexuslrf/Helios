echo "=== Helios SDEdit (stage2 DMD) Test ==="

echo "car: Lambo -> Anime"
python infer_sdedit_helios_stage2.py \
    --video_path "example/car.mp4" \
    --prompt "An anime style Lamborghini Huracan Tecnica car speeds along a curving mountain road, surrounded by lush green trees under a partly cloudy sky." \
    --height 384 --width 640 --num_frames 33 \
    --guidance_scale 1.0 \
    --pyramid_num_stages 3 \
    --pyramid_num_inference_steps_list 4 4 4 \
    --image_path "example/lambo_manga.png" \
    --edit_stage 1.0 \
    --output_folder "./output_helios/distilled_sdedit_i2v/lambo_anime" 