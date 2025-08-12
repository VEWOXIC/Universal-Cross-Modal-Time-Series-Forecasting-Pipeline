for output_len in 24 168
do
python -u filter.py \
    --data Jena_Atmospheric_Physics \
    --output_len $output_len | tee -a ./logs/sampling_info.log

done