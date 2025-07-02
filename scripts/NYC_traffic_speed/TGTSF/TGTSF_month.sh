python -u run.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS_hetero_TGTSF.yaml' \
    --ahead month \
    --batch_size 256 \
    --num_workers 32 | tee ./logs/NYC_traffic_speed/TGTSF_month.log
    