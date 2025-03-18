python -u run.py \
    --model 'iTransformer' \
    --model_config 'model_configs/general/iTransformer.yaml' \
    --data solar \
    --data_config './data_configs/fullsolar.yaml' \
    --ahead day \
    --batch_size 1024 | tee ./logs/solar/iTransformer_day.log
    