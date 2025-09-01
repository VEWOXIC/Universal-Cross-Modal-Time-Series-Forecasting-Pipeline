# python -u run_fm.py \
#     --model 'ChatTime' \
#     --model_config 'model_configs/FM/ChatTime.yaml' \
#     --data California_ISO \
#     --data_config './data_configs/California_ISO/fullCAISO_H.yaml' \
#     --task 'TSF' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --hf_mirror True \
#     --filtered_samples "sample_indexes/California_ISO_sample_day.json" \
#     --gpu 5 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'ChatTime' \
#     --model_config 'model_configs/FM/ChatTime.yaml' \
#     --data California_ISO \
#     --data_config './data_configs/California_ISO/fullCAISO_H.yaml' \
#     --task 'TSF' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --hf_mirror True \
#     --filtered_samples "sample_indexes/California_ISO_sample_week.json" \
#     --gpu 5 | tee -a ./logs/test_FM_on_samples.log

python -u run_fm.py \
    --model 'ChatTime' \
    --model_config 'model_configs/FM/ChatTime.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO_hetero_LLM.yaml' \
    --task 'TGTSF' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --hf_mirror True \
    --filtered_samples "sample_indexes/California_ISO_sample_day.json" \
    --gpu 6 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'ChatTime' \
#     --model_config 'model_configs/FM/ChatTime.yaml' \
#     --data California_ISO \
#     --data_config './data_configs/California_ISO/fullCAISO_hetero_LLM.yaml' \
#     --task 'TGTSF' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --hf_mirror True \
#     --filtered_samples "sample_indexes/California_ISO_sample_week.json" \
#     --gpu 6 | tee -a ./logs/test_FM_on_samples.log