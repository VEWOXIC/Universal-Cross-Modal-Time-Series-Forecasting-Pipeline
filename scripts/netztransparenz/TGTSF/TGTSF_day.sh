python -u run.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data netztransparenz \
    --data_config './data_configs/netztransparenz_hetero_emb.yaml' \
    --ahead day \
    --batch_size 1024 \
    --num_workers 4  | tee ./logs/netztransparenz/TGTSF_day.log
    