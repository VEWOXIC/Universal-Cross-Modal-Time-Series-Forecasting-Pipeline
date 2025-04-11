python -u run.py \
    --model 'PatchTST' \
    --model_config 'model_configs/general/PatchTST.yaml' \
    --data solar \
    --data_config './data_configs/fullsolar.yaml' \
    --ahead week \
    --batch_size 1024 | tee ./logs/solar/PatchTST_week.log

