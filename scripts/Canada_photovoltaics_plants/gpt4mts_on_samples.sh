python -u criterias.py \
    --data 'Canada_photovoltaics_plants' \
    --model 'GPT4MTS' \
    --task 'TGTSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --device "5" \
    --filtered_samples "sample_indexes/Canada_photovoltaics_plants_sample_day.json" | tee -a ./logs/test_RPLLM_on_samples.log

python -u criterias.py \
    --data 'Canada_photovoltaics_plants' \
    --model 'GPT4MTS' \
    --task 'TGTSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --device "5" \
    --filtered_samples "sample_indexes/Canada_photovoltaics_plants_sample_week.json" | tee -a ./logs/test_RPLLM_on_samples.log