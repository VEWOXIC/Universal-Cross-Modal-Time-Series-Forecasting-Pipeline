for output_len in 12 144 288 576
do
python -u criterias.py \
    --model 'GPT4MTS' \
    --data Bear_room \
    --data_config './data_configs/Bear_room/fullBear_hetero_zero_shot_RPLLM.yaml' \
    --input_len 288 \
    --output_len $output_len \
    --device 0 \
    --task "TGTSF" | tee -a ./logs/zero_shot_weather1.log

done