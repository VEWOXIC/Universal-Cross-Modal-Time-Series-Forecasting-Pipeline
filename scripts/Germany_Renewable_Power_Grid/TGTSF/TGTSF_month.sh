python -u run.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_hetero_TGTSF.yaml' \
    --ahead month \
    --batch_size 256 \
    --num_workers 32 | tee ./logs/Germany_Renewable_Power_Grid/TGTSF_month.log
    