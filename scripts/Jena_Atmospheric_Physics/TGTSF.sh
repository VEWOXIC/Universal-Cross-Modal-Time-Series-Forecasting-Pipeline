for output_len in 24 168 # 336 720
do
python -u run_lightning.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data Jena_Atmospheric_Physics \
    --data_config './data_configs/Jena_Atmospheric_Physics/fullJAP_hetero_TGTSF_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 32 \
    --patience 1 \
    --train_epochs 20 \
    --devices '0,1,4' | tee -a ./logs/IATSF_abl.log

done