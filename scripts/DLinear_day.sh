python -u run.py \
    --model 'DLinear' \
    --model_config 'model_configs/general/DLinear.yaml' \
    --data weather \
    --data_config './data_configs/weather.yaml' \
    --input_len 288 \
    --output_len 96 \
    --num_workers 16 \
    --batch_size 1024 \
    --learning_rate 0.0001 #| tee ./logs/traffic/PatchTST_day.log
    