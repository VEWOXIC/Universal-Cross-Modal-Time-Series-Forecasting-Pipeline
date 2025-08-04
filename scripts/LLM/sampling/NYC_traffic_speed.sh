for output_len in 24 168
do
python -u filter_without_inference.py \
    --data NYC_traffic_speed \
    --output_len $output_len \
    --sampling_rate 0.01 | tee -a ./logs/sampling_info.log

done