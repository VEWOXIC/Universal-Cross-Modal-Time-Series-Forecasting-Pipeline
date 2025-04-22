python -u run.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data netztransparenz \
    --data_config './data_configs/netztransparenz_hetero_emb.yaml' \
    --ahead week \
    --batch_size 512 \
    --num_workers 4  | tee ./logs/netztransparenz/TGTSF_week.log