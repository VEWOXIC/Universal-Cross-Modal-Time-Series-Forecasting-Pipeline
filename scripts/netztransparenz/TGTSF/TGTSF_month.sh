python -u run.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data netztransparenz \
    --data_config './data_configs/netztransparenz_hetero_emb.yaml' \
    --ahead month \
    --batch_size 256 \
    --num_workers 32 | tee ./logs/netztransparenz/TGTSF_month.log
    