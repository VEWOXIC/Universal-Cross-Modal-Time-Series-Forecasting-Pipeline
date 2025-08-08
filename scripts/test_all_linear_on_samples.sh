python -u criterias.py \
    --data 'Canada_photovoltaics_plants' \
    --model 'DLinear' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --device "cuda:2" \
    --evaluate_mode "all_samples" \
    --filtered_samples "sample_indexes/Canada_photovoltaics_plants_sample_day.json" | tee -a ./logs/test_linear_on_samples.log

python -u criterias.py \
    --data 'Canada_photovoltaics_plants' \
    --model 'DLinear' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --device "cuda:2" \
    --evaluate_mode "all_samples" \
    --filtered_samples "sample_indexes/Canada_photovoltaics_plants_sample_week.json" | tee -a ./logs/test_linear_on_samples.log

python -u criterias.py \
    --data 'Canada_photovoltaics_plants' \
    --model 'FITS' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --device "cuda:2" \
    --evaluate_mode "all_samples" \
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
    --evaluate_mode "all_samples" \
    --filtered_samples "sample_indexes/Canada_photovoltaics_plants_sample_week.json" | tee -a ./logs/test_linear_on_samples.log

python -u criterias.py \
    --data 'Germany_Renewable_Power_Grid' \
    --model 'DLinear' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --device "cuda:2" \
    --evaluate_mode "all_samples" \
    --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_day.json" | tee -a ./logs/test_linear_on_samples.log

python -u criterias.py \
    --data 'Germany_Renewable_Power_Grid' \
    --model 'DLinear' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --device "cuda:2" \
    --evaluate_mode "all_samples" \
    --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_week.json" | tee -a ./logs/test_linear_on_samples.log

python -u criterias.py \
    --data 'Germany_Renewable_Power_Grid' \
    --model 'FITS' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --device "cuda:2" \
    --evaluate_mode "all_samples" \
    --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_day.json" | tee -a ./logs/test_linear_on_samples.log

python -u criterias.py \
    --data 'Germany_Renewable_Power_Grid' \
    --model 'FITS' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --device "cuda:2" \
    --evaluate_mode "all_samples" \
    --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_week.json" | tee -a ./logs/test_linear_on_samples.log

python -u criterias.py \
    --data 'Jena_Atmospheric_Physics' \
    --model 'DLinear' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --device "cuda:2" \
    --evaluate_mode "all_samples" \
    --filtered_samples "sample_indexes/Jena_Atmospheric_Physics_sample_day.json" | tee -a ./logs/test_linear_on_samples.log

python -u criterias.py \
    --data 'Jena_Atmospheric_Physics' \
    --model 'DLinear' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --device "cuda:2" \
    --evaluate_mode "all_samples" \
    --filtered_samples "sample_indexes/Jena_Atmospheric_Physics_sample_week.json" | tee -a ./logs/test_linear_on_samples.log

python -u criterias.py \
    --data 'Jena_Atmospheric_Physics' \
    --model 'FITS' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --device "cuda:2" \
    --evaluate_mode "all_samples" \
    --filtered_samples "sample_indexes/Jena_Atmospheric_Physics_sample_day.json" | tee -a ./logs/test_linear_on_samples.log

python -u criterias.py \
    --data 'Jena_Atmospheric_Physics' \
    --model 'FITS' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --device "cuda:2" \
    --evaluate_mode "all_samples" \
    --filtered_samples "sample_indexes/Jena_Atmospheric_Physics_sample_week.json" | tee -a ./logs/test_linear_on_samples.log

python -u criterias.py \
    --data 'NYC_traffic_speed' \
    --model 'DLinear' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --device "cuda:2" \
    --evaluate_mode "all_samples" \
    --filtered_samples "sample_indexes/NYC_traffic_speed_sample_day.json" | tee -a ./logs/test_linear_on_samples.log

python -u criterias.py \
    --data 'NYC_traffic_speed' \
    --model 'DLinear' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --device "cuda:2" \
    --evaluate_mode "all_samples" \
    --filtered_samples "sample_indexes/NYC_traffic_speed_sample_week.json" | tee -a ./logs/test_linear_on_samples.log

python -u criterias.py \
    --data 'NYC_traffic_speed' \
    --model 'FITS' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --device "cuda:2" \
    --evaluate_mode "all_samples" \
    --filtered_samples "sample_indexes/NYC_traffic_speed_sample_day.json" | tee -a ./logs/test_linear_on_samples.log

python -u criterias.py \
    --data 'NYC_traffic_speed' \
    --model 'FITS' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --device "cuda:2" \
    --evaluate_mode "all_samples" \
    --filtered_samples "sample_indexes/NYC_traffic_speed_sample_week.json" | tee -a ./logs/test_linear_on_samples.log

python -u criterias.py \
    --data 'California_ISO' \
    --model 'DLinear' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --device "cuda:3" \
    --evaluate_mode "all_samples" \
    --filtered_samples "sample_indexes/California_ISO_sample_day.json" | tee -a ./logs/test_linear_on_samples.log

python -u criterias.py \
    --data 'California_ISO' \
    --model 'DLinear' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --device "cuda:3" \
    --evaluate_mode "all_samples" \
    --filtered_samples "sample_indexes/California_ISO_sample_week.json" | tee -a ./logs/test_linear_on_samples.log

python -u criterias.py \
    --data 'California_ISO' \
    --model 'FITS' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --device "cuda:3" \
    --evaluate_mode "all_samples" \
    --filtered_samples "sample_indexes/California_ISO_sample_day.json" | tee -a ./logs/test_linear_on_samples.log

python -u criterias.py \
    --data 'California_ISO' \
    --model 'FITS' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --device "cuda:3" \
    --evaluate_mode "all_samples" \
    --filtered_samples "sample_indexes/California_ISO_sample_week.json" | tee -a ./logs/test_linear_on_samples.log