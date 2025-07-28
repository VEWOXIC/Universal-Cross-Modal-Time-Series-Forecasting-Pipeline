for output_len in 144 # 12 144 288 576
do
python -u run_lightning.py \
    --model 'PatchTST' \
    --model_config 'model_configs/general/PatchTST.yaml' \
    --data Bear_room \
    --data_config './data_configs/Bear_room/fullBear_H.yaml' \
    --input_len 288 \
    --output_len $output_len \
    --train_epochs 50 \
    --batch_size 64 | tee -a ./logs/Trans.log

done