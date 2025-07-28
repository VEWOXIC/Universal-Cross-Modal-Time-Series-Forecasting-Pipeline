for output_len in 24 168 336 720
do
python -u run_fm.py \
    --model 'Chronos' \
    --model_config 'model_configs/FM/Chronos.yaml' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 128 \
    --gpu 2 | tee -a ./logs/FM.log
    
done