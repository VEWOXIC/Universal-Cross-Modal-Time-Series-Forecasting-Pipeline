python -u run_llm.py \
    --model 'qwen2.5-14b-instruct-1m' \
    --model_config './model_configs/LLM/UniModal/Qwen2.5-14B-Instruct-1m.yaml' \
    --data Jena_Atmospheric_Physics \
    --data_config './data_configs/Jena_Atmospheric_Physics/fullJAP_hetero_LLM.yaml' \
    --input_len 360 \
    --output_len 168 \
    --checkpoints ./checkpoints \
    --filtered_samples ./sample_indexes/Jena_Atmospheric_Physics_sample_week.json | tee -a ./logs/1M_uni_Jena_week.log
    # --sample_step 12 \
    # --no_parallel
    