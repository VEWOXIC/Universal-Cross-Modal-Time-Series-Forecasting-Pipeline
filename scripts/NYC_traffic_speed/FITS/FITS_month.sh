python -u run.py \
    --model 'FITS' \
    --model_config 'model_configs/general/FITS.yaml' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS.yaml' \
    --ahead month \
    --batch_size 1024 | tee ./logs/NYC_traffic_speed/FITS_month.log
    