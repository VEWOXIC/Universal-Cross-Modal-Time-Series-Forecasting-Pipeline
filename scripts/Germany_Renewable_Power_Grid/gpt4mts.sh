for output_len in 24 168 336 720
do
python -u run.py \
    --model 'GPT4MTS' \
    --model_config 'model_configs/general/GPT4MTS.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_hetero_RPLLM_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --hf_offline True \
    --gpu 4 \
    --batch_size 256 | tee -a ./logs/RPLLM1.log
    
done