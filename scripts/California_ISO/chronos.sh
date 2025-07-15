for output_len in 24 168 336 720
do
python -u fm_run.py \
    --model 'Chronos' \
    --model_config 'model_configs/FM/Chronos.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 64 \
    --gpu 2 | tee -a ./logs/FM.log
    
done