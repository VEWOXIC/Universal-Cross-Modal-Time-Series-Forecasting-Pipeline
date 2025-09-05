python -u run_llm.py \
    --model 'deepseek-r1-250120' \
    --model_config './model_configs/LLM/Unimodel/DeepSeek-R1.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_hetero_LLM.yaml' \
    --input_len 360 \
    --output_len 168 \
    --checkpoints ./checkpoints \
    --filtered_samples ./sample_indexes/Germany_Renewable_Power_Grid_sample_week.json | tee -a ./logs/DS_uni_Germany_week.log
    # --sample_step 12 \
    # --no_parallel
    