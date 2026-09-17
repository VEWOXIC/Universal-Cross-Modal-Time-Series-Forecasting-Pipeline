for output_len in 576 12 144 288
do
python -u run_fm.py \
    --model 'Sundial' \
    --model_config 'model_configs/FM/Sundial.yaml' \
    --data Bear_room \
    --data_config data_configs/Bear_room/Bear.yaml \
    --input_len 288 \
    --output_len $output_len \
    --batch_size 128 \
    --gpu 0 | tee -a ./logs/FM.log
    
done