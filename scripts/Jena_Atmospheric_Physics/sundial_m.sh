for output_len in 144 1008
do
python -u fm_run.py \
    --model 'Sundial' \
    --model_config 'model_configs/FM/Sundial.yaml' \
    --data Jena_Atmospheric_Physics \
    --data_config './data_configs/Jena_Atmospheric_Physics/fullJAP.yaml' \
    --input_len 2160 \
    --output_len $output_len \
    --batch_size 64 \
    --gpu 3 | tee -a ./logs/FM.log
    
done