for output_len in 720 336 168 24
do
python -u run_fm.py \
    --model 'Chronos' \
    --model_config 'model_configs/FM/Chronos.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 128 \
    --gpu 0 | tee -a ./logs/FM.log
    
done