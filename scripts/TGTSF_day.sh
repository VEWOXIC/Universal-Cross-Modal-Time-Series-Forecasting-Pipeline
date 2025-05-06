export CUDA_VISIBLE_DEVICES=1,2,3

python -u run_lightning.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF-weather.yaml' \
    --data weather \
    --data_config './data_configs/weather_hetero_emb.yaml' \
    --input_len 288 \
    --output_len 96 \
    --batch_size 112 \
    --num_workers 16 \
    --use_multi_gpu \
    --devices 0,1,2 | tee ./logs/weather/TGTSF_day.log
    