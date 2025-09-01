python -u criterias_lightning.py \
    --data 'Germany_Renewable_Power_Grid' \
    --baseline_model 'TGTSF' \
    --task 'TGTSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --device "3" \
    --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_day.json" | tee -a ./logs/test_IATSF_on_samples.log

python -u criterias_lightning.py \
    --data 'Germany_Renewable_Power_Grid' \
    --baseline_model 'TGTSF' \
    --task 'TGTSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --device "3" \
    --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_week.json" | tee -a ./logs/test_IATSF_on_samples.log