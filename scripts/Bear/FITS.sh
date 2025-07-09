for output_len in 24 168 336
do
python -u run.py \
    --model 'FITS' \
    --model_config 'model_configs/general/FITS.yaml' \
    --data Bear \
    --data_config './data_configs/Bear/fullBear_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 512 | tee -a ./logs/Linear.log
    
done