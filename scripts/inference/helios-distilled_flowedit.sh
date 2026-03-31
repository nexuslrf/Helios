echo "=== Helios FlowEdit (stage2 DMD) Test ==="

EDIT_TYPE="${1:-flowedit}"   # flowedit | flowalign

echo "car: Lambo -> Porsche"
python infer_flowedit_helios_stage2.py \
    --video_path "example/car.mp4" \
    --target_prompt "A silver Porsche 911 car speeds along a curving mountain road, surrounded by lush green trees under a partly cloudy sky." \
    --source_prompt "A green Lamborghini Huracn Tecnica car speeds along a curving mountain road, surrounded by lush green trees under a partly cloudy sky." \
    --height 384 --width 640 --num_frames 33 \
    --pyramid_num_stages 3 \
    --pyramid_num_inference_steps_list 4 4 4 \
    --edit_type ${EDIT_TYPE} \
    --edit_stage 1.0 \
    --output_folder "./output_helios/distilled_flowedit/lambo_porsche"