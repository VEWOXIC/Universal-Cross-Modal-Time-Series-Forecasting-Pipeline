if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/traffic" ]; then
    mkdir ./logs/traffic
fi

python -u run.py \
    --model 'FITS' \
    --model_config 'model_configs/general/FITS.yaml' \
    --data traffic \
    --data_config './data_configs/fulltraffic.yaml' \
    --ahead day \
    --batch_size 1024 | tee ./logs/traffic/FITS_day.log
    