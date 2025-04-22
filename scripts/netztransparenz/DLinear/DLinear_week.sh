python -u run.py \
    --model 'DLinear' \
    --model_config 'model_configs/general/DLinear.yaml' \
    --data snetztransparenzlar \
    --data_config './data_configs/netztransparenz.yaml' \
    --ahead week \
    --batch_size 1024 | tee ./logs/netztransparenz/DLinear_week.log

