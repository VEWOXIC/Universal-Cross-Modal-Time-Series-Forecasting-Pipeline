for output_len in 24 168 336 720
do
python -u run_lightning.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data Canada_photovoltaics_plants \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP_hetero_TGTSF.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 256 \
    --patience 5 \
    --train_epochs 50 \
    --devices '0,1' #| tee -a ./logs/IATSF.log

done