python run.py \
    --model 'FITS' \
    --model_config 'model_configs/general/FITS.yaml' \
    --data solar \
    --data_config './data_configs/fullsolar.yaml' \
    --input_len 2688 \
    --output_len 672 \
    --batch_size 1024 | tee -a ./logs/solar/FITS_month.log
    