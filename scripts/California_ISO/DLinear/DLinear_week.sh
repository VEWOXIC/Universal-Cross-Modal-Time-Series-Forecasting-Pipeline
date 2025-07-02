python -u run.py \
    --model 'DLinear' \
    --model_config 'model_configs/general/DLinear.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO.yaml' \
    --ahead week \
    --batch_size 1024 | tee ./logs/California_ISO/DLinear_week.log

