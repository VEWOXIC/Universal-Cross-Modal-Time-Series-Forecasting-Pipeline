python -u run.py \
    --model 'PatchTST' \
    --model_config 'model_configs/general/PatchTST.yaml' \
    --data Canada_photovoltaics_plants \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP.yaml' \
    --ahead day \
    --batch_size 1024 | tee ./logs/Canada_photovoltaics_plants/PatchTST_day.log
    