for output_len in 12 144
do
python -u filter.py \
    --data Bear_room \
    --version oldest \
    --sampling_rate 0.05 \
    --input_len 288 \
    --output_len $output_len | tee -a ./logs/sampling_info.log

done