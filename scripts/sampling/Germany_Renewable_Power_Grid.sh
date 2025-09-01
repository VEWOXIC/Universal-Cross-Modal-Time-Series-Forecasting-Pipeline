for output_len in 24 168
do
python -u filter_without_inference.py \
    --data Germany_Renewable_Power_Grid \
    --output_len $output_len | tee -a ./logs/sampling_info.log

done