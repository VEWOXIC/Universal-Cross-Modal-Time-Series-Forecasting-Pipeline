python -u criterias.py \
    --data 'California_ISO' \
    --model 'DLinear' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --device "3" \
    --filtered_samples "sample_indexes/California_ISO_sample_day.json" | tee -a ./logs/test_linear_on_samples.log

python -u criterias.py \
    --data 'California_ISO' \
    --model 'DLinear' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --device "3" \
    --filtered_samples "sample_indexes/California_ISO_sample_week.json" | tee -a ./logs/test_linear_on_samples.log