python -u run.py \
    --model 'PatchTST' \
    --model_config 'model_configs/general/PatchTST.yaml' \
    --data solar \
    --data_config './data_configs/fulltraffic.yaml' \
    --ahead month \
    --batch_size 1024 | tee ./logs/solar/PatchTST_month.log
    