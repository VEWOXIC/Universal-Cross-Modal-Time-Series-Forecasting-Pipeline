python -u llm_run.py \
    --model 'deepseek-r1-250120' \
    --model_config './model_configs/LLM/DeepSeek-R1.yaml' \
    --data solar \
    --data_config './data_configs/fullsolar_hetero.yaml' \
    --ahead day \
    --checkpoints ./checkpoints \
    --filtered_samples ./solar_sample_day.json \
    --sample_step 12 \
    # --no_parallel
    #'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'