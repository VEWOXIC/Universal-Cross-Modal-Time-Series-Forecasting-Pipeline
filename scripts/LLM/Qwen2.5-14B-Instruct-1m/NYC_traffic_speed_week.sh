python -u llm_run.py \
    --model 'qwen2.5-14b-instruct-1m' \
    --model_config './model_configs/LLM/Qwen2.5-14B-Instruct-1m.yaml' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS_hetero_LLM.yaml' \
    --input_len 360 \
    --output_len 168 \
    --checkpoints ./checkpoints \
    --filtered_samples ./sample_indexes/NYC_traffic_speed_sample_week.json | tee -a ./logs/Qwen2.5_14B_1M_Traffic_week.log
    # --sample_step 12 \
    # --no_parallel
    