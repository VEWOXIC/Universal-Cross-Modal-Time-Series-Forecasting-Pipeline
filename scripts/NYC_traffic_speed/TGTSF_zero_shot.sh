for output_len in 24 168 336 720
do
python -u criterias_lightning.py \
    --baseline_model 'TGTSF' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS_zero_shot_hetero_TGTSF_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --task "TGTSF" \
    --device "cuda:0" | tee -a ./logs/zero_shot.log

done