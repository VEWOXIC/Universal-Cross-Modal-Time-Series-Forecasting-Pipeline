python -u criterias.py \
    --data 'Bear_room' \
    --model 'GPT4MTS' \
    --task 'MTSF' \
    --version '08-28-1731' \
    --input_len 288 \
    --output_len 12 \
    --batch_size 1 \
    --device "6" \
    --filtered_samples "sample_indexes/Bear_room_sample_hour.json" | tee -a ./logs/test_RPLLM_on_samples_2.log

python -u criterias.py \
    --data 'Bear_room' \
    --model 'GPT4MTS' \
    --task 'MTSF' \
    --version '08-30-2019' \
    --input_len 288 \
    --output_len 144 \
    --batch_size 1 \
    --device "6" \
    --filtered_samples "sample_indexes/Bear_room_sample_half_a_day.json" | tee -a ./logs/test_RPLLM_on_samples_2.log