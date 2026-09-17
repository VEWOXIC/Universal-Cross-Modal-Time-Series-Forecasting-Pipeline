python -u run_llm.py \
    --model 'qwen2.5-14b-instruct-1m' \
    --model_config './model_configs/LLM/Unimodel/Qwen2.5-14B-Instruct-1m.yaml' \
    --data Canada_photovoltaics_plants \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP_hetero_LLM.yaml' \
    --input_len 360 \
    --output_len 168 \
    --checkpoints ./checkpoints \
    --filtered_samples ./sample_indexes/Canada_photovoltaics_plants_sample_week.json | tee -a ./logs/Qwen2.5_14B_1m_uni_Canada_week.log
    # --sample_step 12 \
    # --no_parallel
    