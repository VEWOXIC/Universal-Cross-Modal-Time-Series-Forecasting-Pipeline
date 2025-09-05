for output_len in 24 168 336 720
do
python -u criterias.py \
    --model 'GPT4MTS' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS_zero_shot_hetero_RPLLM_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --task "TGTSF" | tee -a ./logs/NYC_zero_shot.log

done