python -u run.py \
    --model 'DLinear' \
    --model_config 'model_configs/general/DLinear.yaml' \
    --data traffic \
    --data_config './data_configs/fulltraffic.yaml' \
    --input_len 288 \
    --output_len 96 \
    --batch_size 128 \
    --learning_rate 0.001 #| tee ./logs/traffic/DLinear_general.log
    