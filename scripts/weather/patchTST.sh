for output_len in 96 192 336 720
do
python -u run.py \
    --model 'PatchTST' \
    --model_config 'model_configs/general/PatchTST.yaml' \
    --data weather \
    --data_config './data_configs/weather_H.yaml' \
    --input_len 288 \
    --output_len $output_len \
    --num_workers 16 \
    --batch_size 1024 | tee ./logs/traffic/PatchTST_ds6_$output_len.log

done