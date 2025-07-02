for output_len in 96 192 336 720
do
python -u run.py \
    --model 'iTransformer' \
    --model_config 'model_configs/general/iTransformer.yaml' \
    --data weather \
    --data_config './data_configs/weather_H.yaml' \
    --input_len 288 \
    --output_len $output_len \
    --num_workers 16 \
    --batch_size 1024 | tee ./logs/traffic/iTransformer_ds6_$output_len.log

done