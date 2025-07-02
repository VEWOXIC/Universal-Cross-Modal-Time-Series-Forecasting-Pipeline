python -u run.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO_hetero_TGTSF_general.yaml' \
    --ahead week \
    --batch_size 512 \
    --num_workers 4  | tee ./logs/California_ISO/TGTSF_week.log