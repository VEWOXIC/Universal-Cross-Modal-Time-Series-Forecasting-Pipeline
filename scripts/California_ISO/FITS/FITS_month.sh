python -u run.py \
    --model 'FITS' \
    --model_config 'model_configs/general/FITS.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO.yaml' \
    --ahead month \
    --batch_size 256 | tee ./logs/California_ISO/FITS_month.log
    