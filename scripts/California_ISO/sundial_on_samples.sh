python -u run_fm.py \
    --model 'Sundial' \
    --model_config 'model_configs/FM/Sundial.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO_H.yaml' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --filtered_samples "sample_indexes/California_ISO_sample_day.json" \
    --gpu 3 | tee -a ./logs/test_FM_on_samples.log

python -u run_fm.py \
    --model 'Sundial' \
    --model_config 'model_configs/FM/Sundial.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO_H.yaml' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --filtered_samples "sample_indexes/California_ISO_sample_week.json" \
    --gpu 3 | tee -a ./logs/test_FM_on_samples.log