for output_len in 24 168 336 720
do
python -u run.py \
    --model 'GPT4TS' \
    --model_config 'model_configs/general/GPT4TS.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --hf_mirror True \
    --gpu 3 \
    --batch_size 512 | tee -a ./logs/Linear2.log
    
done