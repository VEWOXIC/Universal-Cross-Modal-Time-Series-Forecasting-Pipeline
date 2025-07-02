python -u run.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_hetero_TGTSF.yaml' \
    --ahead day \
    --batch_size 1024 \
    --num_workers 4  | tee ./logs/Germany_Renewable_Power_Grid/TGTSF_day.log
    