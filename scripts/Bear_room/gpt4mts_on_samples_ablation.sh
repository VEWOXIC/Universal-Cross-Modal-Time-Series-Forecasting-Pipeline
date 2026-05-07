python -u criterias.py \
    --data 'Bear_room' \
    --model 'GPT4MTS' \
    --task 'TGTSF' \
    --version '08-28-1703' \
    --input_len 288 \
    --output_len 12 \
    --batch_size 1 \
    --device "0" \
    --filtered_samples "sample_indexes/Bear_room_sample_hour.json" | tee -a ./logs/test_RPLLM_on_samples_ab.log

python -u criterias.py \
    --data 'Bear_room' \
    --model 'GPT4MTS' \
    --task 'TGTSF' \
    --version '08-29-1245' \
    --input_len 288 \
    --output_len 144 \
    --batch_size 1 \
    --device "0" \
    --filtered_samples "sample_indexes/Bear_room_sample_half_a_day.json" | tee -a ./logs/test_RPLLM_on_samples_ab.log