python -u run_llm.py \
    --model 'qwen2.5-14b-instruct' \
    --model_config './model_configs/LLM/Qwen2.5-14B-Instruct.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_hetero_LLM.yaml' \
    --input_len 360 \
    --output_len 24 \
    --checkpoints ./checkpoints \
    --filtered_samples ./sample_indexes/Germany_Renewable_Power_Grid_sample_day.json | tee -a ./logs/Qwen2.5_14B_Germany_day.log
    # --sample_step 12 \
    # --no_parallel
    