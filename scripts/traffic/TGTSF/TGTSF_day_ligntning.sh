export CUDA_VISIBLE_DEVICES=2,3

python -u run_lightning.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data traffic \
    --data_config './data_configs/fulltraffic_hetero_emb_dbg.yaml' \
    --ahead day \
    --batch_size 1024 \
    --num_workers 8 \
    --use_multi_gpu \
    --devices 0,1 #| tee ./logs/traffic/TGTSF_day.log
    