for output_len in 24 168
do
python -u filter.py \
    --data Canada_photovoltaics_plants \
    --output_len $output_len

done