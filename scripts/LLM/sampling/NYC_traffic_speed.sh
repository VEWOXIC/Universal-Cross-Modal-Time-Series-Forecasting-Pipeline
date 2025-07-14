for output_len in 24 168
do
python -u filter.py \
    --data NYC_traffic_speed \
    --output_len $output_len

done