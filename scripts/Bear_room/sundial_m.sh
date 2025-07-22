for output_len in 288 2016
do
python -u run_fm.py \
    --model 'Sundial' \
    --model_config 'model_configs/FM/Sundial.yaml' \
    --data Bear_room \
    --data_config './data_configs/Bear_room/fullBear.yaml' \
    --input_len 4320 \
    --output_len $output_len \
    --batch_size 64 \
    --gpu 3 | tee -a ./logs/FM.log
    
done