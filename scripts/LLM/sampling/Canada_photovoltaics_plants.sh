for output_len in 24 168
do
python -u filter_without_inference.py \
    --data Canada_photovoltaics_plants \
    --output_len $output_len | tee -a ./logs/sampling_info.log

done