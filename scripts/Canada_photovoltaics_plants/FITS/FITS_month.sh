python -u run.py \
    --model 'FITS' \
    --model_config 'model_configs/general/FITS.yaml' \
    --data Canada_photovoltaics_plants \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP.yaml' \
    --ahead month \
    --batch_size 1024 | tee ./logs/Canada_photovoltaics_plants/FITS_month.log
    