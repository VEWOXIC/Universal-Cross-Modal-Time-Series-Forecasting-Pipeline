python -m memory_profiler run.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data solar \
    --data_config './data_configs/fullsolar_hetero_emb.yaml' \
    --ahead day \
    --batch_size 1024 \
    --num_workers 1 #64 #| tee ./logs/solar/TGTSF_day.log


#--model 'TGTSF' --model_config 'model_configs/general/TGTSF.yaml' --data solar --data_config './data_configs/fullsolar_hetero_emb.yaml' --ahead month --batch_size 64 --num_workers 1 
    