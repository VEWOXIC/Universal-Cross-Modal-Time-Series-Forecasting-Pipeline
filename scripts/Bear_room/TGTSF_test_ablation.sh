for output_len in 288  # 12 144 288 576
do
python -u criterias_lightning.py \
    --baseline_model 'TGTSF' \
    --data Bear_room \
    --input_len 288 \
    --output_len $output_len \
    --version "ligntning_20250809_02" \
    --data_config './data_configs/Bear_room/fullBear_hetero_ablation_TGTSF.yaml' \
    --batch_size 512 \
    --device "cuda:4" \
    --channel_wise True \
    --task "TGTSF" | tee -a ./logs/ablation.log
done

for output_len in 288  # 12 144 288 576
do
python -u criterias_lightning.py \
    --data 'Bear_room' \
    --baseline_model 'iTransformer' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 288 \
    --output_len $output_len \
    --batch_size 512 \
    --channel_wise True \
    --device "cuda:2" | tee -a ./logs/ablation.log
done