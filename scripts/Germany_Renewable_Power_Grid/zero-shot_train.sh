for output_len in 24 168 #336 720
do
python -u run.py \
    --model 'PatchTST' \
    --model_config 'model_configs/general/PatchTST.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_H_obs.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 512 | tee -a ./logs/GRPGzero.log

done

for output_len in 24 168 #336 720
do
python -u run.py \
    --model 'iTransformer' \
    --model_config 'model_configs/general/iTransformer.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_H_obs.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 512 | tee -a ./logs/GRPGzero.log

done

for output_len in 24 168 #336 720
do
python -u run.py \
    --model 'DLinear' \
    --model_config 'model_configs/general/DLinear.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_H_obs.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 512 | tee -a ./logs/GRPGzero.log
    
done

for output_len in 24 168 #336 720
do
python -u run.py \
    --model 'FITS' \
    --model_config 'model_configs/general/FITS.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_H_obs.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 512 | tee -a ./logs/GRPGzero.log
    
done

for output_len in 24 168 #336 720
do
python -u run.py \
    --model 'GPT4TS' \
    --model_config 'model_configs/general/GPT4TS.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_H_obs.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 512 | tee -a ./logs/GRPGzero.log
    # --hf_mirror True \
    
done