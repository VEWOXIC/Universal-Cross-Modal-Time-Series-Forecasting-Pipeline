python -u llm_run.py \
    --model 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B' \
    --model_config './model_configs/LLM/DeepSeek-R1-Distill-Qwen-14B.yaml' \
    --data netztransparenz \
    --data_config './data_configs/netztransparenz_hetero.yaml' \
    --ahead week \
    --checkpoints /data/Blob_WestJP/v-zhijianxu/Reasoning_baselines/checkpoints \
    --filtered_samples ./netztransparenz_sample_week.json \
    --sample_step 12 \
    # --no_parallel
    