python -u run_llm.py \
    --model 'qwen2.5-14b-instruct-1m' \
    --model_config './model_configs/LLM/MultiModal/Qwen2.5-14B-Instruct-1m.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO_hetero_LLM.yaml' \
    --input_len 360 \
    --output_len 168 \
    --checkpoints ./checkpoints \
    --filtered_samples ./sample_indexes/California_ISO_sample_week.json # | tee -a ./logs/Qwen2.5_14B_1M_ISO_week.log
    # --sample_step 12 \
    # --no_parallel
    