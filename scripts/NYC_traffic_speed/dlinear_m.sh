for output_len in 288 2016 4032 8640
do
python -u run.py \
    --model 'DLinear' \
    --model_config 'model_configs/general/DLinear.yaml' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS.yaml' \
    --input_len 4320 \
    --output_len $output_len \
    --batch_size 512 | tee -a ./logs/Linear.log
    
done