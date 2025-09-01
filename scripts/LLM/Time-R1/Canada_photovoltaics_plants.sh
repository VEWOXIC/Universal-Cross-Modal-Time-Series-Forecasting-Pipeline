python -u run_llm.py \
    --model Time-R1 \
    --model_config 'model_configs/LLM/UniModal/Time-R1.yaml' \
    --data Canada_photovoltaics_plants \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP_hetero_LLM.yaml' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --filtered_samples "sample_indexes/Canada_photovoltaics_plants_sample_day.json" # | tee -a ./logs/Time-R1.log

python -u run_llm.py \
    --model Time-R1 \
    --model_config 'model_configs/LLM/UniModal/Time-R1.yaml' \
    --data Canada_photovoltaics_plants \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP_hetero_LLM.yaml' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --filtered_samples "sample_indexes/Canada_photovoltaics_plants_sample_week.json" # | tee -a ./logs/Time-R1.log
