for output_len in 24 72 120 168
do
python -u run_fm.py \
    --model 'TimeMoE' \
    --model_config 'model_configs/FM/TimeMoE.yaml' \
    --data Bear_room \
    --data_config './data_configs/Bear_room/fullBear_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 256 \
    --gpu 3 | tee -a ./logs/FM.log
    
done