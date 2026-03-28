python -u criterias_lightning.py \
    --data 'Canada_photovoltaics_plants' \
    --data_config ./data_configs/Canada_photovoltaics_plants/fullCPP_hetero_TGTSF.yaml \
    --baseline_model 'TGTSF' \
    --task 'TGTSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --device "3" \
    --filtered_samples "sample_random_indexes/Canada_photovoltaics_plants_sample_day.json" | tee -a ./logs/test_IATSF_on_samples.log

python -u criterias_lightning.py \
    --data 'Canada_photovoltaics_plants' \
    --baseline_model 'TGTSF' \
    --task 'TGTSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --device "3" \
    --filtered_samples "sample_random_indexes/Canada_photovoltaics_plants_sample_week.json" | tee -a ./logs/test_IATSF_on_samples.log