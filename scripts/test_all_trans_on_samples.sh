# python -u criterias_lightning.py \
#     --data 'Canada_photovoltaics_plants' \
#     --baseline_model 'iTransformer' \
#     --task 'TSF' \
#     --version 'latest' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --device "cuda:2" \
#     --filtered_samples "sample_indexes/Canada_photovoltaics_plants_sample_day.json" | tee -a ./logs/test_trans_on_samples.log

# python -u criterias_lightning.py \
#     --data 'Canada_photovoltaics_plants' \
#     --baseline_model 'iTransformer' \
#     --task 'TSF' \
#     --version 'latest' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --device "cuda:2" \
#     --filtered_samples "sample_indexes/Canada_photovoltaics_plants_sample_week.json" | tee -a ./logs/test_trans_on_samples.log

# python -u criterias_lightning.py \
#     --data 'Canada_photovoltaics_plants' \
#     --baseline_model 'PatchTST' \
#     --task 'TSF' \
#     --version 'latest' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --device "cuda:2" \
#     --filtered_samples "sample_indexes/Canada_photovoltaics_plants_sample_day.json" | tee -a ./logs/test_trans_on_samples.log

# python -u criterias_lightning.py \
#     --data 'Canada_photovoltaics_plants' \
#     --baseline_model 'PatchTST' \
#     --task 'TSF' \
#     --version 'latest' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --device "cuda:2" \
#     --filtered_samples "sample_indexes/Canada_photovoltaics_plants_sample_week.json" | tee -a ./logs/test_trans_on_samples.log

# python -u criterias_lightning.py \
#     --data 'Germany_Renewable_Power_Grid' \
#     --baseline_model 'iTransformer' \
#     --task 'TSF' \
#     --version 'latest' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --device "cuda:2" \
#     --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_day.json" | tee -a ./logs/test_trans_on_samples.log

# python -u criterias_lightning.py \
#     --data 'Germany_Renewable_Power_Grid' \
#     --baseline_model 'iTransformer' \
#     --task 'TSF' \
#     --version 'latest' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --device "cuda:2" \
#     --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_week.json" | tee -a ./logs/test_trans_on_samples.log

# python -u criterias_lightning.py \
#     --data 'Germany_Renewable_Power_Grid' \
#     --baseline_model 'PatchTST' \
#     --task 'TSF' \
#     --version 'latest' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --device "cuda:2" \
#     --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_day.json" | tee -a ./logs/test_trans_on_samples.log

# python -u criterias_lightning.py \
#     --data 'Germany_Renewable_Power_Grid' \
#     --baseline_model 'PatchTST' \
#     --task 'TSF' \
#     --version 'latest' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --device "cuda:2" \
#     --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_week.json" | tee -a ./logs/test_trans_on_samples.log

# python -u criterias_lightning.py \
#     --data 'Jena_Atmospheric_Physics' \
#     --baseline_model 'iTransformer' \
#     --task 'TSF' \
#     --version 'latest' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --device "cuda:2" \
#     --filtered_samples "sample_indexes/Jena_Atmospheric_Physics_sample_day.json" | tee -a ./logs/test_trans_on_samples.log

# python -u criterias_lightning.py \
#     --data 'Jena_Atmospheric_Physics' \
#     --baseline_model 'iTransformer' \
#     --task 'TSF' \
#     --version 'latest' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --device "cuda:2" \
#     --filtered_samples "sample_indexes/Jena_Atmospheric_Physics_sample_week.json" | tee -a ./logs/test_trans_on_samples.log

# python -u criterias_lightning.py \
#     --data 'Jena_Atmospheric_Physics' \
#     --baseline_model 'PatchTST' \
#     --task 'TSF' \
#     --version 'latest' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --device "cuda:2" \
#     --filtered_samples "sample_indexes/Jena_Atmospheric_Physics_sample_day.json" | tee -a ./logs/test_trans_on_samples.log

# python -u criterias_lightning.py \
#     --data 'Jena_Atmospheric_Physics' \
#     --baseline_model 'PatchTST' \
#     --task 'TSF' \
#     --version 'latest' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --device "cuda:2" \
#     --filtered_samples "sample_indexes/Jena_Atmospheric_Physics_sample_week.json" | tee -a ./logs/test_trans_on_samples.log

python -u criterias_lightning.py \
    --data 'NYC_traffic_speed' \
    --baseline_model 'iTransformer' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --device "cuda:2" \
    --filtered_samples "sample_indexes/NYC_traffic_speed_sample_day.json" | tee -a ./logs/test_trans_on_samples.log

python -u criterias_lightning.py \
    --data 'NYC_traffic_speed' \
    --baseline_model 'iTransformer' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --device "cuda:2" \
    --filtered_samples "sample_indexes/NYC_traffic_speed_sample_week.json" | tee -a ./logs/test_trans_on_samples.log

python -u criterias_lightning.py \
    --data 'NYC_traffic_speed' \
    --baseline_model 'PatchTST' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --device "cuda:2" \
    --filtered_samples "sample_indexes/NYC_traffic_speed_sample_day.json" | tee -a ./logs/test_trans_on_samples.log

python -u criterias_lightning.py \
    --data 'NYC_traffic_speed' \
    --baseline_model 'PatchTST' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --device "cuda:2" \
    --filtered_samples "sample_indexes/NYC_traffic_speed_sample_week.json" | tee -a ./logs/test_trans_on_samples.log

# python -u criterias_lightning.py \
#     --data 'California_ISO' \
#     --baseline_model 'iTransformer' \
#     --task 'TSF' \
#     --version 'latest' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --device "cuda:2" \
#     --filtered_samples "sample_indexes/California_ISO_sample_day.json" | tee -a ./logs/test_trans_on_samples.log

# python -u criterias_lightning.py \
#     --data 'California_ISO' \
#     --baseline_model 'iTransformer' \
#     --task 'TSF' \
#     --version 'latest' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --device "cuda:2" \
#     --filtered_samples "sample_indexes/California_ISO_sample_week.json" | tee -a ./logs/test_trans_on_samples.log

# python -u criterias_lightning.py \
#     --data 'California_ISO' \
#     --baseline_model 'PatchTST' \
#     --task 'TSF' \
#     --version 'latest' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --device "cuda:2" \
#     --filtered_samples "sample_indexes/California_ISO_sample_day.json" | tee -a ./logs/test_trans_on_samples.log

# python -u criterias_lightning.py \
#     --data 'California_ISO' \
#     --baseline_model 'PatchTST' \
#     --task 'TSF' \
#     --version 'latest' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --device "cuda:2" \
#     --filtered_samples "sample_indexes/California_ISO_sample_week.json" | tee -a ./logs/test_trans_on_samples.log