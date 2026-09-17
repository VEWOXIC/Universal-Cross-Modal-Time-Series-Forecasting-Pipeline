for output_len in 576
do
python -u run_fm.py \
    --model 'TimeMoE' \
    --model_config 'model_configs/FM/TimeMoE.yaml' \
    --data Bear_room \
    --data_config data_configs/Bear_room/Bear.yaml \
    --input_len 288 \
    --output_len $output_len \
    --batch_size 64 \
    --gpu 0 | tee -a ./logs/timemoe.log
    
done