# for output_len in 12
# do
# python -u criterias_lightning.py \
#     --baseline_model 'TGTSF' \
#     --data Bear_room \
#     --input_len 288 \
#     --output_len $output_len \
#     --version "oldest" \
#     --channel_wise False \
#     --task "TGTSF" | tee -a ./logs/ablation.log
# done


# python -u criterias_lightning.py \
#     --data 'Canada_photovoltaics_plants' \
#     --baseline_model 'PatchTST' \
#     --task 'TSF' \
#     --version 'latest' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --device "cuda:2"

python -u criterias.py \
    --data 'Canada_photovoltaics_plants' \
    --model 'DLinear' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --device "cuda:2" \