python -u run.py \
    --model 'PatchTST' \
    --model_config 'model_configs/general/PatchTST.yaml' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS.yaml' \
    --ahead day \
    --batch_size 1024 #| tee ./logs/NYC_traffic_speed/PatchTST_day.log
    