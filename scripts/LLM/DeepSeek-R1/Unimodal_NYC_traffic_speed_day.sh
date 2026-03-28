python -u run_llm.py \
    --model 'deepseek-r1-250120' \
    --model_config './model_configs/LLM/DeepSeek-R1.yaml' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS_hetero_LLM.yaml' \
    --input_len 360 \
    --output_len 24 \
    --checkpoints ./checkpoints \
    --filtered_samples ./sample_random_indexes/NYC_traffic_speed_sample_day.json | tee -a ./logs/umLLM.log
    # --sample_step 12 \
    # --no_parallel
    