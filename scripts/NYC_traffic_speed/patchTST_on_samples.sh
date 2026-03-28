python -u criterias_lightning.py \
    --data 'NYC_traffic_speed' \
    --baseline_model 'PatchTST' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --device "2" \
    --filtered_samples "sample_random_indexes/NYC_traffic_speed_sample_day.json" | tee -a ./logs/test_trans_on_samples.log

python -u criterias_lightning.py \
    --data 'NYC_traffic_speed' \
    --baseline_model 'PatchTST' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --device "2" \
    --filtered_samples "sample_random_indexes/NYC_traffic_speed_sample_week.json" | tee -a ./logs/test_trans_on_samples.log