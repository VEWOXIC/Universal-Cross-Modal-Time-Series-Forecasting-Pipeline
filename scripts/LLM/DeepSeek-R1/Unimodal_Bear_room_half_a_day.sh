python -u run_llm.py \
    --model 'deepseek-r1-250120' \
    --model_config './model_configs/LLM/UniModal/DeepSeek-R1.yaml' \
    --data Bear_room \
    --data_config './data_configs/Bear_room/fullBear_hetero_LLM.yaml' \
    --input_len 288 \
    --output_len 144 \
    --checkpoints ./checkpoints \
    --filtered_samples ./sample_indexes/Bear_room_sample_half_a_day.json \
    