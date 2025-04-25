python -u run.py \
    --model 'iTransformer' \
    --model_config 'model_configs/general/iTransformer.yaml' \
    --data traffic \
    --data_config './data_configs/fulltraffic.yaml' \
    --ahead month \
    --batch_size 128 | tee ./logs/traffic/iTransformer_month.log
    