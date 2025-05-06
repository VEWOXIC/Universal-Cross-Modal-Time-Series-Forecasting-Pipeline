python -u run.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF-solar.yaml' \
    --data solar \
    --data_config './data_configs/fullsolar_hetero_emb.yaml' \
    --ahead week \
    --batch_size 256 \
    --num_workers 32 | tee ./logs/solar/TGTSF_week.log
    