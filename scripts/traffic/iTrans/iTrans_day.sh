python -u run.py \
    --model 'iTransformer' \
    --model_config 'model_configs/general/iTransformer.yaml' \
    --data traffic \
    --data_config './data_configs/fulltraffic.yaml' \
    --ahead day \
    --batch_size 1024 | tee ./logs/traffic/iTransformer_day.log
    