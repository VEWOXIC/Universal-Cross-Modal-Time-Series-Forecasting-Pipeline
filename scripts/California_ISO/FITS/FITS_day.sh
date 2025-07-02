python -u run.py \
    --model 'FITS' \
    --model_config 'model_configs/general/FITS.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO.yaml' \
    --ahead day \
    --batch_size 1024 | tee ./logs/California_ISO/FITS_day.log
    