python -u run_llm.py \
    --model 'qwen3-14b' \
    --model_config './model_configs/LLM/Qwen3-14B.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_hetero_LLM.yaml' \
    --input_len 360 \
    --output_len 168 \
    --checkpoints ./checkpoints \
    --filtered_samples ./sample_indexes/Germany_Renewable_Power_Grid_sample_week.json | tee -a ./logs/Qwen3_14B_Germany_week.log
    # --sample_step 12 \
    # --no_parallel
    