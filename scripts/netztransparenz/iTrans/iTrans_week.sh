python -u run.py \
    --model 'iTransformer' \
    --model_config 'model_configs/general/iTransformer.yaml' \
    --data netztransparenz \
    --data_config './data_configs/netztransparenz.yaml' \
    --ahead week \
    --batch_size 1024 | tee ./logs/netztransparenz/iTransformer_week.log

