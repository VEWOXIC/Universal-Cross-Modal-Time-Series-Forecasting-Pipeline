python -u run.py \
    --model 'FITS' \
    --model_config 'model_configs/general/FITS.yaml' \
    --data solar \
    --data_config './data_configs/fullsolar.yaml' \
    --ahead day \
    --batch_size 1024 | tee ./logs/solar/FITS_day.log
    