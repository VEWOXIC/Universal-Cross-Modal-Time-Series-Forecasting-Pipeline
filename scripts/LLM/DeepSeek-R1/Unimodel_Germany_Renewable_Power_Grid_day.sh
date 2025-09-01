python -u llm_run.py \
    --model 'deepseek-r1-250120' \
    --model_config './model_configs/LLM/Unimodel/DeepSeek-R1.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_hetero_LLM.yaml' \
    --input_len 360 \
    --output_len 24 \
    --checkpoints ./checkpoints \
    --filtered_samples ./sample_indexes/Germany_Renewable_Power_Grid_sample_day.json | tee -a ./logs/DS_uni_Germany_day.log
    # --sample_step 12 \
    # --no_parallel
    