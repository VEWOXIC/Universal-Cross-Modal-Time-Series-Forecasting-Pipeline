for output_len in 12 144 288 576
do
python -u criterias_lightning.py \
    --data 'Bear_room' \
    --data_config './data_configs/Bear_room/fullBear_hetero_TGTSF.yaml' \
    --baseline_model 'TGTSF' \
    --task 'TGTSF' \
    --version 'oldest' \
    --input_len 288 \
    --output_len $output_len \
    --batch_size 512 \
    --device "cuda:3" | tee -a ./logs/test_IATSF.log
done