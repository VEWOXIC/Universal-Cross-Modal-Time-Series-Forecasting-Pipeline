python -u run.py \
    --model 'PatchTST' \
    --model_config 'model_configs/general/PatchTST.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG.yaml' \
    --ahead day \
    --batch_size 1024 | tee ./logs/Germany_Renewable_Power_Grid/PatchTST_day.log
    