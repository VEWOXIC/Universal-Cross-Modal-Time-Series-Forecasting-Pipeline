for output_len in 24 168 # 336 720
do
python -u run_fm.py \
    --model 'TimeMoE' \
    --model_config 'model_configs/FM/TimeMoE.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 200 \
    --gpu 0 | tee -a ./logs/FM2.log
    
done