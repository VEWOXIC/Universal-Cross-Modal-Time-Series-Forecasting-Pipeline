python -u run_fm.py \
    --model 'TimeMoE' \
    --model_config 'model_configs/FM/TimeMoE.yaml' \
    --data Bear_room \
    --data_config './data_configs/Bear_room/fullBear.yaml' \
    --input_len 288 \
    --output_len 12 \
    --batch_size 1 \
    --filtered_samples "sample_indexes/Bear_room_sample_hour.json" \
    --gpu 3 | tee -a ./logs/test_FM_on_samples.log

python -u run_fm.py \
    --model 'TimeMoE' \
    --model_config 'model_configs/FM/TimeMoE.yaml' \
    --data Bear_room \
    --data_config './data_configs/Bear_room/fullBear.yaml' \
    --input_len 288 \
    --output_len 144 \
    --batch_size 1 \
    --filtered_samples "sample_indexes/Bear_room_sample_half_a_day.json" \
    --gpu 3 | tee -a ./logs/test_FM_on_samples.log