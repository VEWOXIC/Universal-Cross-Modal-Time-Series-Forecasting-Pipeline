for output_len in 12 144 288 576
do
python -u run.py \
    --model 'GPT4TS' \
    --model_config 'model_configs/general/GPT4TS.yaml' \
    --data Bear_room \
    --data_config './data_configs/Bear_room/fullBear.yaml' \
    --input_len 288 \
    --output_len $output_len \
    --hf_mirror True \
    --gpu 3 \
    --batch_size 256 | tee -a ./logs/Linear2.log
    
done