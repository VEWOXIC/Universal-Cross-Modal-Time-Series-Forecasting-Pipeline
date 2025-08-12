python -u run_fm.py \
    --model 'Chronos' \
    --model_config 'model_configs/FM/Chronos.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_H.yaml' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_day.json" \
    --gpu 1 | tee -a ./logs/test_FM_on_samples.log

python -u run_fm.py \
    --model 'Chronos' \
    --model_config 'model_configs/FM/Chronos.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_H.yaml' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_week.json" \
    --gpu 1 | tee -a ./logs/test_FM_on_samples.log