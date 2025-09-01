for output_len in 288 2016 4032 8640
do
python -u run_lightning.py \
    --model 'iTransformer' \
    --model_config 'model_configs/general/iTransformer.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO.yaml' \
    --input_len 4320 \
    --output_len $output_len \
    --batch_size 512 | tee -a ./logs/Trans.log

done