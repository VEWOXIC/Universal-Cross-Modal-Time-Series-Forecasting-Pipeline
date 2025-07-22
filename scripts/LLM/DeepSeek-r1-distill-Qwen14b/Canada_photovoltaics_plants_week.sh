python -u run_llm.py \
    --model 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B' \
    --model_config './model_configs/LLM/DeepSeek-R1-Distill-Qwen-14B.yaml' \
    --data solar \
    --data_config './data_configs/fullsolar_hetero.yaml' \
    --ahead week \
    --checkpoints ./checkpoints \
    --filtered_samples ./sample_indexes/solar_sample_week.json \
    --sample_step 24 
    