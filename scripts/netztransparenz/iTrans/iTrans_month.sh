python -u run.py \
    --model 'iTransformer' \
    --model_config 'model_configs/general/iTransformer.yaml' \
    --data netztransparenz \
    --data_config './data_configs/netztransparenz.yaml' \
    --ahead month \
    --batch_size 256 | tee ./logs/netztransparenz/iTransformer_month.log
    