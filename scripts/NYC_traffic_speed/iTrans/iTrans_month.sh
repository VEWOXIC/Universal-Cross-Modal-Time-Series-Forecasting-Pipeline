python -u run.py \
    --model 'iTransformer' \
    --model_config 'model_configs/general/iTransformer.yaml' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS.yaml' \
    --ahead month \
    --batch_size 128 | tee ./logs/NYC_traffic_speed/iTransformer_month.log
    