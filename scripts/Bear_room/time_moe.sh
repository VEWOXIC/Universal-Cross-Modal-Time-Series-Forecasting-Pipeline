for output_len in 12 144 288 576
do
python -u run_fm.py \
    --model 'TimeMoE' \
    --model_config 'model_configs/FM/TimeMoE.yaml' \
    --data Bear_room \
    --data_config './data_configs/Bear_room/fullBear.yaml' \
    --input_len 288 \
    --output_len $output_len \
    --batch_size 128 \
    --gpu 3 | tee -a ./logs/FM.log
    
done