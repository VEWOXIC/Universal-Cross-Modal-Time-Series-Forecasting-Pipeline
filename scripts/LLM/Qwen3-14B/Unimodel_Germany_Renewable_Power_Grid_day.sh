python -u llm_run.py \
    --model 'qwen3-14b' \
    --model_config './model_configs/LLM/Unimodel/Qwen3-14B.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_hetero_LLM.yaml' \
    --input_len 360 \
    --output_len 24 \
    --checkpoints ./checkpoints \
    --filtered_samples ./sample_indexes/Germany_Renewable_Power_Grid_sample_day.json | tee -a ./logs/Qwen3_14B_uni_Germany_day.log
    # --sample_step 12 \
    # --no_parallel
    