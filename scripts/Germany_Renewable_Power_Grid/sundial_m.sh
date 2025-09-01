for output_len in 96 672
do
python -u run_fm.py \
    --model 'Sundial' \
    --model_config 'model_configs/FM/Sundial.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG.yaml' \
    --input_len 1440 \
    --output_len $output_len \
    --batch_size 128 \
    --gpu 1 | tee -a ./logs/FM.log
    
done