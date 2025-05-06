python -u run.py \
    --model 'PatchTST' \
    --model_config 'model_configs/general/PatchTST.yaml' \
    --data weather \
    --data_config './data_configs/weather.yaml' \
    --input_len 288 \
    --output_len 96 \
    --batch_size 256 #| tee ./logs/weather/PatchTST_96.log
    