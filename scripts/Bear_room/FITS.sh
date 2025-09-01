for output_len in 12 144 288 576
do
python -u run.py \
    --model 'FITS' \
    --model_config 'model_configs/general/FITS.yaml' \
    --data Bear_room \
    --data_config './data_configs/Bear_room/fullBear.yaml' \
    --input_len 288 \
    --output_len $output_len \
    --batch_size 128 \
    --patience 5 \
    --train_epochs 50 \
    --devices '2' | tee -a ./logs/Linear.log
    
done