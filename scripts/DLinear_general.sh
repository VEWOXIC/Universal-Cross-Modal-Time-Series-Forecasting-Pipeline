python -u run.py \
    --model 'DLinear' \
    --model_config 'model_configs/general/DLinear.yaml' \
    --data ETT \
    --data_config './data_configs/fullETT.yaml' \
    --input_len 288 \
    --output_len 96 \
    --batch_size 256 \
    --learning_rate 0.0005 #| tee ./logs/traffic/DLinear_general.log
    