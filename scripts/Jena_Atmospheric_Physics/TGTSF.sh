for output_len in 24 168 336 720
do
python -u run.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data Jena_Atmospheric_Physics \
    --data_config './data_configs/Jena_Atmospheric_Physics/fullJAP_hetero_TGTSF_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 512 | tee ./logs/Linear.log

done