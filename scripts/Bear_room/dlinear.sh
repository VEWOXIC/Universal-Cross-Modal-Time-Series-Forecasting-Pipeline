for output_len in 12 144 288 576
do
python -u run.py \
    --model 'DLinear' \
    --model_config 'model_configs/general/DLinear.yaml' \
    --data Bear_room \
    --data_config './data_configs/Bear_room/fullBear.yaml' \
    --input_len 288 \
    --output_len $output_len \
    --batch_size 256 \
    --devices '2' | tee -a ./logs/Linear.log
    
done