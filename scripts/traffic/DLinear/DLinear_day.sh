python -u run.py \
    --model 'DLinear' \
    --model_config 'model_configs/general/DLinear.yaml' \
    --data traffic \
    --data_config './data_configs/fulltraffic.yaml' \
    --ahead day \
    --input_len 96 \
    --output_len 96 \
    --batch_size 128 | tee ./logs/traffic/DLinear_day.log \
    