python -u llm_run.py \
    --model 'deepseek-v3-250324' \
    --model_config './model_configs/LLM/DeepSeek-V3.yaml' \
    --data solar \
    --data_config './data_configs/fullsolar_hetero.yaml' \
    --ahead day \
    --checkpoints ./checkpoints \
    --filtered_samples ./sample_indexes/solar_sample_day.json \
    --sample_step 12 \
    --no_parallel \