python -u llm_run.py \
    --model 'qwen2.5-14b-instruct-1m' \
    --model_config './model_configs/LLM/Unimodel/Qwen2.5-14B-Instruct-1m.yaml' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS_hetero_LLM.yaml' \
    --input_len 360 \
    --output_len 24 \
    --checkpoints ./checkpoints \
    --filtered_samples ./sample_indexes/NYC_traffic_speed_sample_day.json | tee -a ./logs/Qwen2.5_14B_1m_uni_Traffic_day.log
    # --sample_step 12 \
    # --no_parallel
    