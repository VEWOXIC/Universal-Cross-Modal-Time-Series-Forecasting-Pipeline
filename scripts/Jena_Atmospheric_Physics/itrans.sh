for output_len in 24 168 336 720
do
python -u run_lightning.py \
    --model 'iTransformer' \
    --model_config 'model_configs/general/iTransformer.yaml' \
    --data Jena_Atmospheric_Physics \
    --data_config './data_configs/Jena_Atmospheric_Physics/fullJAP_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 128 | tee -a ./logs/Trans.log

done