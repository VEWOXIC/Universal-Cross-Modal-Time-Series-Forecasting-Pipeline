export CUDA_VISIBLE_DEVICES=1,2,3

python -u run_lightning.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS_hetero_TGTSF.yaml' \
    --ahead day \
    --batch_size 1024 \
    --num_workers 16 \
    --use_multi_gpu \
    --devices 0,1,2 #| tee ./logs/NYC_traffic_speed/TGTSF_day.log
    