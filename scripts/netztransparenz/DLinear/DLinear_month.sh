python -u run.py \
    --model 'DLinear' \
    --model_config 'model_configs/general/DLinear.yaml' \
    --data netztransparenz \
    --data_config './data_configs/netztransparenz.yaml' \
    --ahead month \
    --batch_size 1024 | tee ./logs/netztransparenz/DLinear_month.log
    