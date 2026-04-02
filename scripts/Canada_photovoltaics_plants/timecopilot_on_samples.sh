python -u run_agent.py \
    --model 'TimeCopilot' \
    --model_config 'model_configs/agent/TimeCopilot.yaml' \
    --data Canada_photovoltaics_plants \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP_obs.yaml' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --filtered_samples "sample_indexes/Canada_photovoltaics_plants_sample_day.json" | tee -a ./logs/test_agent_on_samples.log

python -u run_agent.py \
    --model 'TimeCopilot' \
    --model_config 'model_configs/agent/TimeCopilot.yaml' \
    --data Canada_photovoltaics_plants \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP_obs.yaml' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --filtered_samples "sample_indexes/Canada_photovoltaics_plants_sample_week.json" \ | tee -a ./logs/test_agent_on_samples.log