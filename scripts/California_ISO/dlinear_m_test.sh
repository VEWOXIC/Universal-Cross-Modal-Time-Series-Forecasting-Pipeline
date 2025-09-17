for output_len in 288 2016  # 4032 8640
do
python -u criterias_lightning.py \
    --data 'California_ISO' \
    --baseline_model 'DLinear' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 4320 \
    --output_len $output_len \
    --batch_size 512 \
    --device "0" | tee -a ./logs/test_m_iso.log
done