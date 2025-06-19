python -u run.py \
    --model 'PatchTST' \
    --model_config 'model_configs/general/PatchTST.yaml' \
    --data traffic \
    --data_config './data_configs/fulltraffic.yaml' \
    --input_len 288 \
    --output_len 96 \
    --downsample 6 \
    --batch_size 128 \
    --learning_rate 0.001 #| tee ./logs/traffic/PatchTST_day.log
    