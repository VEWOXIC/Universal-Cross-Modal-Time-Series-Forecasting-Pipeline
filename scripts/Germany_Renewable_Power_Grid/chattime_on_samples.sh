# python -u run_fm.py \
#     --model 'ChatTime' \
#     --model_config 'model_configs/FM/ChatTime.yaml' \
#     --data Germany_Renewable_Power_Grid \
#     --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_H.yaml' \
#     --task 'TSF' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --hf_mirror True \
#     --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_day.json" \
#     --gpu 2 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'ChatTime' \
#     --model_config 'model_configs/FM/ChatTime.yaml' \
#     --data Germany_Renewable_Power_Grid \
#     --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_H.yaml' \
#     --task 'TSF' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --hf_mirror True \
#     --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_week.json" \
#     --gpu 2 | tee -a ./logs/test_FM_on_samples.log

python -u run_fm.py \
    --model 'ChatTime' \
    --model_config 'model_configs/FM/ChatTime.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_hetero_LLM.yaml' \
    --task 'TGTSF' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --hf_mirror True \
    --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_day.json" \
    --gpu 0 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'ChatTime' \
#     --model_config 'model_configs/FM/ChatTime.yaml' \
#     --data Germany_Renewable_Power_Grid \
#     --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_hetero_LLM.yaml' \
#     --task 'TGTSF' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --hf_mirror True \
#     --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_week.json" \
#     --gpu 0 | tee -a ./logs/test_FM_on_samples.log