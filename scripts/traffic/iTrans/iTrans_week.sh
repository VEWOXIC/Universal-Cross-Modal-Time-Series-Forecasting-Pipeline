python -u run.py \
    --model 'iTransformer' \
    --model_config 'model_configs/general/iTransformer.yaml' \
    --data traffic \
    --data_config './data_configs/fulltraffic.yaml' \
    --ahead week \
    --batch_size 256 | tee ./logs/traffic/iTransformer_week.log

