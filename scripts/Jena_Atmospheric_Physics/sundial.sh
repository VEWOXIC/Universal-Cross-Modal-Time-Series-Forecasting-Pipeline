for output_len in 24 168 # 336 720
do
python -u run_fm.py \
    --model 'Sundial' \
    --model_config 'model_configs/FM/Sundial.yaml' \
    --data Jena_Atmospheric_Physics \
    --data_config './data_configs/Jena_Atmospheric_Physics/fullJAP_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 1024 \
    --gpu 0 | tee -a ./logs/FM2.log
    
done