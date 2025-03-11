python -u run.py \
    --model 'DLinear' \
    --model_config 'model_configs/general/DLinear.yaml' \
    --data solar \
    --data_config './data_configs/fullsolar.yaml' \
    --ahead month \
    --batch_size 1024 | tee ./logs/solar/DLinear_month.log
    