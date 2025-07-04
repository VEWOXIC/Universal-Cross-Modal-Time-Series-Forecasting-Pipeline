for output_len in 24 168 336 720
do
python -u run.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS_hetero_TGTSF_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 512 | tee ./logs/Linear.log

done