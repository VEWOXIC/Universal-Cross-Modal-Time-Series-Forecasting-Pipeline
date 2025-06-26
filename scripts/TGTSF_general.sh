# export CUDA_VISIBLE_DEVICES=0

for noise in 0.0
do

# python -u run_lightning.py \
python -u run.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF-CAISO.yaml' \
    --data CAISO \
    --data_config './data_configs/fullCAISO_hetero_emb.yaml' \
    --input_len 288 \
    --output_len 96 \
    --batch_size 256 \
    --noise $noise \
    --patience 10 \
    --learning_rate 0.001 \
    --train_epochs 1 \
    # --devices 0,2,3 | tee ./logs/weather/TGTSF_96_$noise.log
    # --num_workers 16 \
    # --use_multi_gpu \
    # --checkpoints '/data/Blob_WestJP/v-zhijianxu/TGTSF_abl/' \
done