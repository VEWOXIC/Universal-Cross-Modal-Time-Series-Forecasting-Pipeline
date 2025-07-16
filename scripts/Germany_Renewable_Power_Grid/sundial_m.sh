for output_len in 96 672
do
python -u fm_run.py \
    --model 'Sundial' \
    --model_config 'model_configs/FM/Sundial.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG.yaml' \
    --input_len 1440 \
    --output_len $output_len \
    --batch_size 64 \
    --gpu 3 | tee -a ./logs/FM.log
    
done