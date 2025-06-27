python -u run.py \
    --model 'DLinear' \
    --model_config 'model_configs/general/DLinear.yaml' \
    --data ETT \
    --data_config './data_configs/fullETT.yaml' \
    --input_len 720 \
    --output_len 96 \
    --batch_size 1024 \
    --learning_rate 0.001 #| tee ./logs/traffic/DLinear_general.log
    