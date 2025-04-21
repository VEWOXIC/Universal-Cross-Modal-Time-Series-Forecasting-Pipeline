export CUDA_VISIBLE_DEVICES=1,2,3

python -u run_lightning.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data traffic \
    --data_config './data_configs/fulltraffic_hetero_emb.yaml' \
    --ahead day \
    --batch_size 1024 \
    --num_workers 16 \
    --use_multi_gpu \
    --test \
    --last_ckpt checkpoints/04-18-0943_TGTSF_traffic_day_ahead_pl/checkpoint-epoch=00-val_loss=0.532713.ckpt \
    --devices 0,1,2 #| tee ./logs/traffic/TGTSF_day.log
    