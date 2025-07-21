for output_len in 24 168 336 720
do
python -u fm_run.py \
    --model 'Chronos' \
    --model_config 'model_configs/FM/Chronos.yaml' \
    --data Jena_Atmospheric_Physics \
    --data_config './data_configs/Jena_Atmospheric_Physics/fullJAP_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 100 \
    --gpu 0 | tee -a ./logs/FM.log
    
done