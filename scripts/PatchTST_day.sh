python -u run.py \
    --model 'PatchTST' \
    --model_config 'model_configs/general/PatchTST.yaml' \
    --data weather \
    --data_config './data_configs/weather.yaml' \
    --input_len 288 \
    --output_len 96 \
    --num_workers 16 \
    --batch_size 1024 #| tee ./logs/traffic/PatchTST_day.log
    