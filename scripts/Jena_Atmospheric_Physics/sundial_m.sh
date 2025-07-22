for output_len in 144 1008
do
python -u run_fm.py \
    --model 'Sundial' \
    --model_config 'model_configs/FM/Sundial.yaml' \
    --data Jena_Atmospheric_Physics \
    --data_config './data_configs/Jena_Atmospheric_Physics/fullJAP.yaml' \
    --input_len 2160 \
    --output_len $output_len \
    --batch_size 128 \
    --gpu 1 | tee -a ./logs/FM.log
    
done