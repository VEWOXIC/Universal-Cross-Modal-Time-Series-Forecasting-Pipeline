for output_len in 24 168 336 720
do
python -u criterias.py \
    --model 'FITS' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS_zero_shot_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --task "TSF" | tee -a ./logs/zero_shot.log

done