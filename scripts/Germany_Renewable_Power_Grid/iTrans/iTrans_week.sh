python -u run.py \
    --model 'iTransformer' \
    --model_config 'model_configs/general/iTransformer.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG.yaml' \
    --ahead week \
    --batch_size 1024 | tee ./logs/Germany_Renewable_Power_Grid/iTransformer_week.log

