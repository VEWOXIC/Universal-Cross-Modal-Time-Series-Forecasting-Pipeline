python -u criterias.py \
    --data 'Canada_photovoltaics_plants' \
    --model 'FITS' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --device "cuda:2" \
    --filtered_samples "sample_indexes/Canada_photovoltaics_plants_sample_day.json" | tee -a ./logs/test_linear_on_samples.log

python -u criterias.py \
    --data 'Canada_photovoltaics_plants' \
    --model 'FITS' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --device "cuda:2" \
    --filtered_samples "sample_indexes/Canada_photovoltaics_plants_sample_week.json" | tee -a ./logs/test_linear_on_samples.log