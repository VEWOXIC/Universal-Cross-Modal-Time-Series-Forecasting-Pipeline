python -u run.py \
    --model 'FITS' \
    --model_config 'model_configs/general/FITS.yaml' \
    --data netztransparenz \
    --data_config './data_configs/netztransparenz.yaml' \
    --ahead month \
    --batch_size 1024 | tee ./logs/netztransparenz/FITS_month.log
    