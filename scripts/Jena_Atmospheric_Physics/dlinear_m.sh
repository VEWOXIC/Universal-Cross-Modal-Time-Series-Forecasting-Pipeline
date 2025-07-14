for output_len in 144 1008 2016 4320
do
python -u run_lightning.py \
    --model 'DLinear' \
    --model_config 'model_configs/general/DLinear.yaml' \
    --data Jena_Atmospheric_Physics \
    --data_config './data_configs/Jena_Atmospheric_Physics/fullJAP.yaml' \
    --input_len 2160 \
    --output_len $output_len \
    --batch_size 256 \
    --device '3' | tee -a ./logs/Linear.log
    
done