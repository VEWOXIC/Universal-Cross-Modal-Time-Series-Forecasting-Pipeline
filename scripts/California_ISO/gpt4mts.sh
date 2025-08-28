for output_len in 24 168 336 720
do
python -u run.py \
    --model 'GPT4MTS' \
    --model_config 'model_configs/general/GPT4MTS.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO_hetero_RPLLM_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --hf_offline True \
    --gpu 5 \
    --batch_size 50 | tee -a ./logs/RPLLM2.log
    
done