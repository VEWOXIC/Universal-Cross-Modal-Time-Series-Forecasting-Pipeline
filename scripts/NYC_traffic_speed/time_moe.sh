for output_len in 24 168 336 720
do
python -u fm_run.py \
    --model 'TimeMoE' \
    --model_config 'model_configs/FM/TimeMoE.yaml' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 512 \
    --gpu 3 | tee -a ./logs/FM.log
    
done