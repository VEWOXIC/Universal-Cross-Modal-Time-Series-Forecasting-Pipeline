python -u run.py \
    --model 'FITS' \
    --model_config 'model_configs/general/FITS.yaml' \
    --data traffic \
    --data_config './data_configs/fulltraffic.yaml' \
    --input_len 90 \
    --output_len 192 \
    --batch_size 64 \
    --learning_rate 0.0005  #| tee ./logs/traffic/FITS_day.log