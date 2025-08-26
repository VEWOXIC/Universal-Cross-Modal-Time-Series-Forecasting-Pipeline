for output_len in 24 168 336 720
do
python -u criterias_lightning.py \
    --data 'NYC_traffic_speed' \
    --baseline_model 'iTransformer' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 512 \
    --device "2" | tee -a ./logs/test_trans.log
done