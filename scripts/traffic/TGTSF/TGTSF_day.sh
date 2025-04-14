python -u run.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data traffic \
    --data_config './data_configs/fulltraffic_hetero_emb.yaml' \
    --ahead day \
    --batch_size 512 \
    --num_workers 16 | tee ./logs/traffic/TGTSF_day.log
    