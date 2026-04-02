python -u run_agent.py \
    --model 'TimeCopilot' \
    --model_config 'model_configs/agent/TimeCopilot.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_H_obs.yaml' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --filtered_samples "sample_indexes/Canada_photovoltaics_plants_sample_day.json" | tee -a ./logs/test_agent_on_samples_2.log

python -u run_agent.py \
    --model 'TimeCopilot' \
    --model_config 'model_configs/agent/TimeCopilot.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_H_obs.yaml' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --filtered_samples "sample_indexes/Canada_photovoltaics_plants_sample_week.json" \ | tee -a ./logs/test_agent_on_samples_2.log