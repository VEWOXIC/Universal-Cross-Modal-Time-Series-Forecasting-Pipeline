python -m memory_profiler run.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data solar \
    --data_config './data_configs/fullsolar_hetero_emb.yaml' \
    --ahead day \
    --batch_size 1024 \
    --num_workers 1 #64 #| tee ./logs/solar/TGTSF_day.log


#--model 'TGTSF' --model_config 'model_configs/general/TGTSF.yaml' --data traffic --data_config './data_configs/fulltraffic_hetero_emb.yaml' --ahead day --batch_size 512 --num_workers 8 
    