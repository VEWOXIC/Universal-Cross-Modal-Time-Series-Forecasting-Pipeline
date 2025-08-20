for output_len in 24 168 336 720
do
python -u run.py \
    --model 'GPT4TS' \
    --model_config 'model_configs/general/GPT4TS.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --hf_mirror True \
    --gpu 3 \
    --batch_size 128 | tee -a ./logs/Linear2.log
    
done