python -u run.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO_hetero_TGTSF_general.yaml' \
    --ahead day \
    --batch_size 1024 \
    --num_workers 4  | tee ./logs/California_ISO/TGTSF_day.log
    