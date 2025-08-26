for output_len in 12 144 288 576
do
python -u criterias_lightning.py \
    --data 'Bear_room' \
    --baseline_model 'iTransformer' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 288 \
    --output_len $output_len \
    --batch_size 512 \
    --device "2" | tee -a ./logs/test_trans.log
done