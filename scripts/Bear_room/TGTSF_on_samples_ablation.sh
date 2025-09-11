python -u criterias_lightning.py \
    --data 'Bear_room' \
    --baseline_model 'TGTSF' \
    --task 'TGTSF' \
    --version 'oldest' \
    --input_len 288 \
    --output_len 12 \
    --batch_size 1 \
    --device "1" \
    --filtered_samples "sample_indexes/Bear_room_sample_hour.json" | tee -a ./logs/test_IATSF_on_samples_ab.log

python -u criterias_lightning.py \
    --data 'Bear_room' \
    --baseline_model 'TGTSF' \
    --task 'TGTSF' \
    --version 'oldest' \
    --input_len 288 \
    --output_len 144 \
    --batch_size 1 \
    --device "1" \
    --filtered_samples "sample_indexes/Bear_room_sample_half_a_day.json" | tee -a ./logs/test_IATSF_on_samples_ab.log