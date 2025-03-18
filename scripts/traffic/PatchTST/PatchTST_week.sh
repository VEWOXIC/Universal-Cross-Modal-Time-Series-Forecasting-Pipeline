python -u run.py \
    --model 'PatchTST' \
    --model_config 'model_configs/general/PatchTST.yaml' \
    --data traffic \
    --data_config './data_configs/fulltraffic.yaml' \
    --ahead week \
    --batch_size 256 | tee ./logs/traffic/PatchTST_week.log

