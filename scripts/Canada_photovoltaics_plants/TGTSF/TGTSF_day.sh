python -u run.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF-solar.yaml' \
    --data Canada_photovoltaics_plants \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP_hetero_TGTSF_general.yaml' \
    --ahead day \
    --batch_size 256 \
    --num_workers 64 | tee ./logs/Canada_photovoltaics_plants/TGTSF_day.log
    