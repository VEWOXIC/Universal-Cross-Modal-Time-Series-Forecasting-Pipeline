for output_len in 24 168
do
python -u filter_random.py \
    --data NYC_traffic_speed \
    --version latest \
    --sampling_rate 0.005 \
    --input_len 360 \
    --output_len $output_len | tee -a ./logs/sampling_random_info.log

done