python -u criterias.py \
    --data 'NYC_traffic_speed' \
    --model 'GPT4MTS' \
    --task 'TGTSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --device "0" \
    --filtered_samples "sample_indexes/NYC_traffic_speed_sample_day.json" | tee -a ./logs/test_RPLLM_on_samples_2.log

python -u criterias.py \
    --data 'NYC_traffic_speed' \
    --model 'GPT4MTS' \
    --task 'TGTSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --device "0" \
    --filtered_samples "sample_indexes/NYC_traffic_speed_sample_week.json" | tee -a ./logs/test_RPLLM_on_samples_2.log