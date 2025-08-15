for output_len in 24 168 336 720
do
python -u criterias_lightning.py \
    --data 'California_ISO' \
    --baseline_model 'PatchTST' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 512 \
    --device "cuda:2" | tee -a ./logs/test_trans.log
done