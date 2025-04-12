python -u run.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data solar \
    --data_config './data_configs/fullsolar_hetero_emb.yaml' \
    --ahead month \
    --batch_size 256 \
    --num_workers 64 | tee ./logs/solar/TGTSF_month.log
    