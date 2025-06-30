python -u run.py \
    --model 'FITS' \
    --model_config 'model_configs/general/FITS.yaml' \
    --data ETTh2 \
    --data_config './data_configs/fullETT_H.yaml' \
    --input_len 90 \
    --output_len 96 \
    --batch_size 256 \
    --learning_rate 0.001  #| tee ./logs/traffic/FITS_day.log