# python -u run_fm.py \
#     --model 'ChatTime' \
#     --model_config 'model_configs/FM/ChatTime.yaml' \
#     --data Jena_Atmospheric_Physics \
#     --data_config './data_configs/Jena_Atmospheric_Physics/fullJAP_H.yaml' \
#     --task 'TGTSF' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --hf_mirror True \
#     --filtered_samples "sample_indexes/Jena_Atmospheric_Physics_sample_day.json" \
#     --gpu 4 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'ChatTime' \
#     --model_config 'model_configs/FM/ChatTime.yaml' \
#     --data Jena_Atmospheric_Physics \
#     --data_config './data_configs/Jena_Atmospheric_Physics/fullJAP_H.yaml' \
#     --task 'TGTSF' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --hf_mirror True \
#     --filtered_samples "sample_indexes/Jena_Atmospheric_Physics_sample_week.json" \
#     --gpu 4 | tee -a ./logs/test_FM_on_samples.log

python -u run_fm.py \
    --model 'ChatTime' \
    --model_config 'model_configs/FM/ChatTime.yaml' \
    --data Jena_Atmospheric_Physics \
    --data_config './data_configs/Jena_Atmospheric_Physics/fullJAP_hetero_LLM.yaml' \
    --task 'TGTSF' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --hf_mirror True \
    --filtered_samples "sample_indexes/Jena_Atmospheric_Physics_sample_day.json" \
    --gpu 4 | tee -a ./logs/test_FM_on_samples.log

python -u run_fm.py \
    --model 'ChatTime' \
    --model_config 'model_configs/FM/ChatTime.yaml' \
    --data Jena_Atmospheric_Physics \
    --data_config './data_configs/Jena_Atmospheric_Physics/fullJAP_hetero_LLM.yaml' \
    --task 'TGTSF' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --hf_mirror True \
    --filtered_samples "sample_indexes/Jena_Atmospheric_Physics_sample_week.json" \
    --gpu 4 | tee -a ./logs/test_FM_on_samples.log