for output_len in 24 168 336 720
do
python -u run_lightning.py \
    --model 'iTransformer' \
    --model_config 'model_configs/general/iTransformer.yaml' \
    --data Bear \
    --data_config './data_configs/Bear/fullBear_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 128 | tee -a ./logs/Trans.log

done