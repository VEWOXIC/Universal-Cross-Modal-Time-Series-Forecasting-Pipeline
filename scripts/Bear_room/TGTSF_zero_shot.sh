for output_len in 12 144 288 576
do
python -u criterias_lightning.py \
    --baseline_model 'TGTSF' \
    --data Bear_room \
    --data_config './data_configs/Bear_room/fullBear_hetero_zero_shot_TGTSF.yaml' \
    --input_len 288 \
    --output_len $output_len \
    --version "ligntning_20250809_0?" \
    --decive 4 \
    --task "TGTSF" | tee -a ./logs/zero_shot_weather.log

done