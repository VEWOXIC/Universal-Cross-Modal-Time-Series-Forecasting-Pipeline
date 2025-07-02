python -u run.py \
    --model 'DLinear' \
    --model_config 'model_configs/general/DLinear.yaml' \
    --data Canada_photovoltaics_plants \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP.yaml' \
    --ahead week \
    --batch_size 1024 | tee ./logs/Canada_photovoltaics_plants/DLinear_week.log

