python -u run.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data traffic \
    --data_config './data_configs/fulltraffic_hetero_emb.yaml' \
    --ahead day \
    --batch_size 1024 \
    --num_workers 4  | tee ./logs/traffic/TGTSF_day.log
    