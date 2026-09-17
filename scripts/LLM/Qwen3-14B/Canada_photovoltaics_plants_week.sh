python -u run_llm.py \
    --model 'qwen3-14b' \
    --model_config './model_configs/LLM/Qwen3-14B.yaml' \
    --data Canada_photovoltaics_plants \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP_hetero_LLM.yaml' \
    --input_len 360 \
    --output_len 168 \
    --checkpoints ./checkpoints \
    --filtered_samples ./sample_indexes/Canada_photovoltaics_plants_sample_week.json | tee -a ./logs/Qwen3_14B_Canada_week.log
    # --sample_step 12 \
    # --no_parallel
    