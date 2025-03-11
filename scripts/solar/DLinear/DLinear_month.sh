python -u run.py \
    --model 'DLinear' \
    --model_config 'model_configs/general/DLinear.yaml' \
    --data solar \
    --data_config './data_configs/fullsolar.yaml' \
    --input_len 2688 \
    --output_len 672 \
    --batch_size 1024 | tee ./logs/solar/DLinear_month.log
    