# for output_len in 288
# do
# python -u criterias_lightning_new.py \
#     --baseline_model 'TGTSF' \
#     --data Bear_room \
#     --input_len 288 \
#     --output_len $output_len \
#     --version "oldest" \
#     --channel_wise False \
#     --data_config './data_configs/Bear_room/fullBear_hetero_TGTSF.yaml' \
#     --batch_size 128 \
#     --device "cuda:3" \
#     --task "TGTSF" | tee -a ./logs/ablation.log
# done

for output_len in 24
do
python -u criterias_lightning_new.py \
    --data 'Canada_photovoltaics_plants' \
    --baseline_model 'iTransformer' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 128 \
    --device "cuda:2" # | tee -a ./logs/test_trans.log
done