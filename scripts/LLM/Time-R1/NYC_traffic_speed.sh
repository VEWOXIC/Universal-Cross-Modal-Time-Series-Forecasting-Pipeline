python -u run_llm.py \
    --model 'qwen3-14b' \
    --model_config './model_configs/LLM/Qwen3-14B.yaml' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS_hetero_LLM.yaml' \
    --input_len 360 \
    --output_len 24 \
    --checkpoints ./checkpoints \
    --filtered_samples ./sample_indexes/NYC_traffic_speed_sample_day.json | tee -a ./logs/Qwen3_14B_Traffic_day.log
    # --sample_step 12 \
    # --no_parallel
    