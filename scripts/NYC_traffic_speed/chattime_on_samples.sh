# python -u run_fm.py \
#     --model 'ChatTime' \
#     --model_config 'model_configs/FM/ChatTime.yaml' \
#     --data NYC_traffic_speed \
#     --data_config './data_configs/NYC_traffic_speed/fullNYCTS_H.yaml' \
#     --task 'TSF' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --hf_mirror True \
#     --filtered_samples "sample_indexes/NYC_traffic_speed_sample_day.json" \
#     --gpu 4 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'ChatTime' \
#     --model_config 'model_configs/FM/ChatTime.yaml' \
#     --data NYC_traffic_speed \
#     --data_config './data_configs/NYC_traffic_speed/fullNYCTS_H.yaml' \
#     --task 'TSF' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --hf_mirror True \
#     --filtered_samples "sample_indexes/NYC_traffic_speed_sample_week.json" \
#     --gpu 4 | tee -a ./logs/test_FM_on_samples.log

python -u run_fm.py \
    --model 'ChatTime' \
    --model_config 'model_configs/FM/ChatTime.yaml' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS_hetero_LLM.yaml' \
    --task 'TGTSF' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --hf_mirror True \
    --filtered_samples "sample_indexes/NYC_traffic_speed_sample_day.json" \
    --gpu 5 | tee -a ./logs/test_FM_on_samples.log

python -u run_fm.py \
    --model 'ChatTime' \
    --model_config 'model_configs/FM/ChatTime.yaml' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS_hetero_LLM.yaml' \
    --task 'TGTSF' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --hf_mirror True \
    --filtered_samples "sample_indexes/NYC_traffic_speed_sample_week.json" \
    --gpu 5 | tee -a ./logs/test_FM_on_samples.log