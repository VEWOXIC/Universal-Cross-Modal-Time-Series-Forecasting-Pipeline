python -u run.py \
    --model 'iTransformer' \
    --model_config 'model_configs/general/iTransformer.yaml' \
    --data Canada_photovoltaics_plants \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP.yaml' \
    --ahead day \
    --batch_size 1024 | tee ./logs/Canada_photovoltaics_plants/iTransformer_day.log
    