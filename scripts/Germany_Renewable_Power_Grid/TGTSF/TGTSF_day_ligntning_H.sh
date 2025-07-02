export CUDA_VISIBLE_DEVICES=1,2,3

python -u run_lightning.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_hetero_TGTSF_H.yaml' \
    --ahead day \
    --batch_size 1024 \
    --num_workers 16 \
    --use_multi_gpu \
    --devices 0,1,2 #| tee ./logs/traffic/TGTSF_day.log
    