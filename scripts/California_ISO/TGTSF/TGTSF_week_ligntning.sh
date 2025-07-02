export CUDA_VISIBLE_DEVICES=1,2,3

python -u run_lightning.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO_hetero_TGTSF_general.yaml' \
    --ahead week \
    --batch_size 512 \
    --num_workers 16 \
    --use_multi_gpu \
    --devices 0,1,2 | tee ./logs/California_ISO/TGTSF_week.log
    