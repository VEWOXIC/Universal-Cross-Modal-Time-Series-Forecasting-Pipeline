python -u llm_run.py \
    --model 'gpt-4.1-nano' \
    --model_config './model_configs/LLM/GPT-4.1-nano.yaml' \
    --data solar \
    --data_config './data_configs/fullsolar_hetero.yaml' \
    --ahead day \
    --checkpoints ./checkpoints \
    --filtered_samples ./sample_indexes/solar_sample_day.json \
    --sample_step 12 \
    --no_parallel \
    #'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'