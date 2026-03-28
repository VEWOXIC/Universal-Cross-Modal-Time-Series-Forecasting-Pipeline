# python -u run_fm.py \
#     --model 'ChatTime' \
#     --model_config 'model_configs/FM/ChatTime.yaml' \
#     --data Bear_room \
#     --data_config './data_configs/Bear_room/fullBear.yaml' \
#     --task 'TSF' \
#     --input_len 288 \
#     --output_len 12 \
#     --batch_size 1 \
#     --hf_mirror True \
#     --filtered_samples "sample_indexes/Bear_room_sample_hour.json" \
#     --gpu 0 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'ChatTime' \
#     --model_config 'model_configs/FM/ChatTime.yaml' \
#     --data Bear_room \
#     --data_config './data_configs/Bear_room/fullBear.yaml' \
#     --task 'TSF' \
#     --input_len 288 \
#     --output_len 144 \
#     --batch_size 1 \
#     --hf_mirror True \
#     --filtered_samples "sample_indexes/Bear_room_sample_half_a_day.json" \
#     --gpu 0 | tee -a ./logs/test_FM_on_samples.log

python -u run_fm.py \
    --model 'ChatTime' \
    --model_config 'model_configs/FM/ChatTime.yaml' \
    --data Bear_room \
    --data_config './data_configs/Bear_room/fullBear_hetero_LLM.yaml' \
    --task 'TGTSF' \
    --input_len 288 \
    --output_len 12 \
    --batch_size 1 \
    --hf_mirror True \
    --filtered_samples "sample_indexes/Bear_room_sample_hour.json" \
    --gpu 1 | tee -a ./logs/test_FM_on_samples_chattime_mm.log

python -u run_fm.py \
    --model 'ChatTime' \
    --model_config 'model_configs/FM/ChatTime.yaml' \
    --data Bear_room \
    --data_config './data_configs/Bear_room/fullBear_hetero_LLM.yaml' \
    --task 'TGTSF' \
    --input_len 288 \
    --output_len 144 \
    --batch_size 1 \
    --hf_mirror True \
    --filtered_samples "sample_indexes/Bear_room_sample_half_a_day.json" \
    --gpu 1 | tee -a ./logs/test_FM_on_samples_chattime_mm.log