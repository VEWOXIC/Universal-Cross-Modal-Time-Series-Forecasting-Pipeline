python -u criterias.py \
    --data 'Bear_room' \
    --model 'GPT4TS' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 288 \
    --output_len 12 \
    --batch_size 1 \
    --device "cuda:3" \
    --filtered_samples "sample_indexes/Bear_room_sample_hour.json" | tee -a ./logs/test_RPLLM_on_samples.log

python -u criterias.py \
    --data 'Bear_room' \
    --model 'GPT4TS' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 288 \
    --output_len 144 \
    --batch_size 1 \
    --device "cuda:3" \
    --filtered_samples "sample_indexes/Bear_room_sample_half_a_day.json" | tee -a ./logs/test_RPLLM_on_samples.log