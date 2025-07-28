for output_len in 144 # 12 144 288 576
do
python -u run_lightning.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF/TGTSF-Bear.yaml' \
    --data Bear_room \
    --data_config './data_configs/Bear_room/fullBear_hetero_TGTSF_H.yaml' \
    --input_len 288 \
    --output_len $output_len \
    --batch_size 64 \
    --patience 10 \
    --train_epochs 50 \
    --devices '0,1,4' | tee -a ./logs/IATSF.log

done