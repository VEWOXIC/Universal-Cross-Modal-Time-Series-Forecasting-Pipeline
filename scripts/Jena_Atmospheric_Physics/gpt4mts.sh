for output_len in 24 168 336 720
do
python -u run.py \
    --model 'GPT4MTS' \
    --model_config 'model_configs/general/GPT4MTS.yaml' \
    --data Jena_Atmospheric_Physics \
    --data_config './data_configs/Jena_Atmospheric_Physics/fullJAP_hetero_RPLLM_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --hf_offline True \
    --gpu 3 \
    --batch_size 50 | tee -a ./logs/RPLLM3.log
    
done