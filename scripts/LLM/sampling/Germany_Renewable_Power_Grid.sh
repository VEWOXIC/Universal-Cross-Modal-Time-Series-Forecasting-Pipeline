for output_len in 24 168
do
python -u filter.py \
    --data Germany_Renewable_Power_Grid \
    --output_len $output_len

done