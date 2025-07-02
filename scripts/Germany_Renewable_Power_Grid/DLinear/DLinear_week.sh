python -u run.py \
    --model 'DLinear' \
    --model_config 'model_configs/general/DLinear.yaml' \
    --data snetztransparenzlar \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG.yaml' \
    --ahead week \
    --batch_size 1024 | tee ./logs/Germany_Renewable_Power_Grid/DLinear_week.log

