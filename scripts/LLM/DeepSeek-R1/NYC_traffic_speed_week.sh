python -u run_llm.py \
    --model 'deepseek-r1-250120' \
    --model_config './model_configs/LLM/DeepSeek-R1.yaml' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS_hetero_LLM.yaml' \
    --input_len 360 \
    --output_len 168 \
    --checkpoints ./checkpoints \
    --filtered_samples ./sample_indexes/NYC_traffic_speed_sample_week.json | tee -a ./logs/DS_Traffic_week.log
    # --sample_step 12 \
    # --no_parallel
    