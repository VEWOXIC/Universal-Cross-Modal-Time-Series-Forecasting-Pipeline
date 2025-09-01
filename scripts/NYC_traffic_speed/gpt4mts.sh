for output_len in 720 # 24 168 336 720
do
python -u run.py \
    --model 'GPT4MTS' \
    --model_config 'model_configs/general/GPT4MTS.yaml' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS_hetero_RPLLM_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --hf_offline True \
    --gpu 4 \
    --batch_size 256 | tee -a ./logs/traffic720.log
    
done