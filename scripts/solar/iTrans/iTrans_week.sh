python -u run.py \
    --model 'iTransformer' \
    --model_config 'model_configs/general/iTransformer.yaml' \
    --data solar \
    --data_config './data_configs/fullsolar.yaml' \
    --ahead week \
    --batch_size 1024 | tee ./logs/solar/iTransformer_week.log

