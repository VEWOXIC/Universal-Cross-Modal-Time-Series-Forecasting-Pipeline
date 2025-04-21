python -u run.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data traffic \
    --data_config './data_configs/fulltraffic_hetero_emb.yaml' \
    --ahead month \
    --batch_size 256 \
    --num_workers 32 | tee ./logs/traffic/TGTSF_month.log
    