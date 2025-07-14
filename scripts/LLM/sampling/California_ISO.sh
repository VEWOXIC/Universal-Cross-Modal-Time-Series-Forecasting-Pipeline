for output_len in 24 168
do
python -u filter.py \
    --data California_ISO \
    --output_len $output_len

done