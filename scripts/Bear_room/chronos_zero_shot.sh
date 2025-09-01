for output_len in 12 144 288 576
do
python -u run_fm.py \
    --model 'Chronos' \
    --model_config 'model_configs/FM/Chronos.yaml' \
    --data Bear_room \
    --data_config './data_configs/Bear_room/fullBear_zero_shot.yaml' \
    --input_len 288 \
    --output_len $output_len \
    --batch_size 512 \
    --gpu 1 | tee -a ./logs/zero_shot.log
    
done