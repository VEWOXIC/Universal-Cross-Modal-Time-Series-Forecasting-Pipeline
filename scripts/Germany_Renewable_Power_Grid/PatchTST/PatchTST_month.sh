python -u run.py \
    --model 'PatchTST' \
    --model_config 'model_configs/general/PatchTST.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG.yaml' \
    --ahead month \
    --batch_size 256 | tee ./logs/Germany_Renewable_Power_Grid/PatchTST_month.log
    