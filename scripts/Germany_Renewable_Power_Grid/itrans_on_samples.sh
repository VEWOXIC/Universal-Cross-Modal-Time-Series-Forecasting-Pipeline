python -u criterias_lightning.py \
    --data 'Germany_Renewable_Power_Grid' \
    --baseline_model 'iTransformer' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --device "2" \
    --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_day.json" | tee -a ./logs/test_trans_on_samples.log

python -u criterias_lightning.py \
    --data 'Germany_Renewable_Power_Grid' \
    --baseline_model 'iTransformer' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --device "2" \
    --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_week.json" | tee -a ./logs/test_trans_on_samples.log