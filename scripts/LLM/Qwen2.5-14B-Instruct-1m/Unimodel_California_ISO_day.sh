python -u llm_run.py \
    --model 'qwen2.5-14b-instruct-1m' \
    --model_config './model_configs/LLM/Unimodel/Qwen2.5-14B-Instruct-1m.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO_hetero_LLM.yaml' \
    --input_len 360 \
    --output_len 24 \
    --checkpoints ./checkpoints \
    --filtered_samples ./sample_indexes/California_ISO_sample_day.json | tee -a ./logs/Qwen2.5_14B_1M_ISO_uni_day.log
    # --sample_step 12 \
    # --no_parallel
    