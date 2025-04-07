python -u llm_run.py \
    --model 'Qwen/QwQ-32B' \
    --model_config './model_configs/LLM/QwQ-32B.yaml' \
    --data solar \
    --data_config './data_configs/fullsolar_hetero.yaml' \
    --input_len 336 \
    --output_len 168 \
    --checkpoints /data/Blob_WestJP/v-zhijianxu/Reasoning_baselines/checkpoints \
    --sample_step 48 \
    