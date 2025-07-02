python -u run.py \
    --model 'PatchTST' \
    --model_config 'model_configs/general/PatchTST.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO.yaml' \
    --ahead month \
    --batch_size 256 | tee ./logs/California_ISO/PatchTST_month.log
    