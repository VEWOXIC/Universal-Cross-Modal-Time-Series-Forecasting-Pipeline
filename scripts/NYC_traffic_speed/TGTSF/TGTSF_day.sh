python -u run.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS_hetero_TGTSF.yaml' \
    --ahead day \
    --batch_size 1024 \
    --num_workers 4  | tee ./logs/NYC_traffic_speed/TGTSF_day.log
    