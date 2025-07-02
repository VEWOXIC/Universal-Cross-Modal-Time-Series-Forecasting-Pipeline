export CUDA_VISIBLE_DEVICES=1,2,3

python -u run_lightning.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO_hetero_TGTSF_general.yaml' \
    --ahead day \
    --batch_size 1024 \
    --num_workers 16 \
    --use_multi_gpu \
    --devices 0,1,2 #| tee ./logs/traffic/TGTSF_day.log
    