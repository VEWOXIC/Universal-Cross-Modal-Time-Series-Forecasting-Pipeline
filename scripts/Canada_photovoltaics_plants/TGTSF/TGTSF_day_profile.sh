python -m memory_profiler run.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data Canada_photovoltaics_plants \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP_hetero_TGTSF_general.yaml' \
    --ahead day \
    --batch_size 1024 \
    --num_workers 1 #64 #| tee ./logs/Canada_photovoltaics_plants/TGTSF_day.log


#--model 'TGTSF' --model_config 'model_configs/general/TGTSF.yaml' --data Canada_photovoltaics_plants --data_config './data_configs/Canada_photovoltaics_plants/fullCPP_hetero_TGTSF_general.yaml' --ahead month --batch_size 64 --num_workers 1 
    