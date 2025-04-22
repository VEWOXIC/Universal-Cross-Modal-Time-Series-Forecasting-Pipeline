python -u run.py \
    --model 'PatchTST' \
    --model_config 'model_configs/general/PatchTST.yaml' \
    --data netztransparenz \
    --data_config './data_configs/netztransparenz.yaml' \
    --ahead month \
    --batch_size 1024 | tee ./logs/netztransparenz/PatchTST_month.log
    