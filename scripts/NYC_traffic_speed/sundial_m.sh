for output_len in 2016
do
python -u fm_run.py \
    --model 'Sundial' \
    --model_config 'model_configs/FM/Sundial.yaml' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS.yaml' \
    --input_len 4320 \
    --output_len $output_len \
    --batch_size 128 \
    --gpu 4 | tee -a ./logs/FM.log
    
done