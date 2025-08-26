for output_len in 24 # 168 336 720
do
python -u run.py \
    --model 'GPT4MTS' \
    --model_config 'model_configs/general/GPT4MTS.yaml' \
    --data Bear_room \
    --data_config './data_configs/Bear_room/fullBear_hetero_LLM.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --hf_offline True \
    --gpu 1 \
    --batch_size 1 # | tee -a ./logs/Linear.log
    
done