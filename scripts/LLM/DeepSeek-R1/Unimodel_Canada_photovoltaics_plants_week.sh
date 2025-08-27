python -u llm_run.py \
    --model 'deepseek-r1-250120' \
    --model_config './model_configs/LLM/Unimodel/DeepSeek-R1.yaml' \
    --data Canada_photovoltaics_plants \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP_hetero_LLM.yaml' \
    --input_len 360 \
    --output_len 168 \
    --checkpoints ./checkpoints \
    --filtered_samples ./sample_indexes/Canada_photovoltaics_plants_sample_week.json | tee -a ./logs/DS_uni_Canada_week.log
    # --sample_step 12 \
    # --no_parallel
    