python -u run.py \
    --model 'DLinear' \
    --model_config 'model_configs/general/DLinear.yaml' \
    --data solar \
    --data_config './data_configs/fulltraffic.yaml' \
    --ahead day \
    --batch_size 1024 | tee ./logs/traffic/DLinear_day.log
    