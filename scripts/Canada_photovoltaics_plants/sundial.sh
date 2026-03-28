for output_len in 24 168 # 336 720
do
python -u run_fm.py \
    --model 'Sundial' \
    --model_config 'model_configs/FM/Sundial.yaml' \
    --data Canada_photovoltaics_plants \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 1024 \
    --gpu 0 | tee -a ./logs/FM2.log
    
done