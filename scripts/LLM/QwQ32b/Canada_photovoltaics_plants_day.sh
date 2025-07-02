python -u llm_run.py \
    --model 'Qwen/QwQ-32B' \
    --model_config './model_configs/LLM/QwQ-32B.yaml' \
    --data solar \
    --data_config './data_configs/fullsolar_hetero.yaml' \
    --ahead day \
    --checkpoints ./checkpoints \
    --filtered_samples ./sample_indexes/solar_sample_day.json \
    --sample_step 12 \
    # --no_parallel
    