for output_len in 24 168 336 720
do
python -u fm_run.py \
    --model 'Chronos' \
    --model_config 'model_configs/FM/Chronos.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 128 \
    --gpu 0 | tee -a ./logs/FM.log
    
done