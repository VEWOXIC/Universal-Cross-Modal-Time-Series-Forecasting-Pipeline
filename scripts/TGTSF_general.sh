# export CUDA_VISIBLE_DEVICES=0

for noise in 0.0
do

python -u run_lightning.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF-weather.yaml' \
    --data weather \
    --data_config './data_configs/weather_hetero_emb.yaml' \
    --input_len 288 \
    --output_len 96 \
    --batch_size 80 \
    --num_workers 16 \
    --use_multi_gpu \
    --noise $noise \
    --patience 10 \
    --checkpoints '/data/Blob_WestJP/v-zhijianxu/TGTSF_abl/' \
    --devices 0,2,3 | tee ./logs/weather/TGTSF_96_$noise.log
    
done