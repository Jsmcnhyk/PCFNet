#!/bin/bash

# create log directories
mkdir -p ./logs/ShortForecasting/PCFNet

model_name=PCFNet
seq_len=96
GPU=0
root=./dataset/  
alpha=0
data_name=PEMS

export CUDA_VISIBLE_DEVICES=${GPU}

########################
# PEMS03
########################
for p in 12
do
python -u run.py \
  --is_training 1 \
  --root_path ${root}/PEMS/ \
  --data_path PEMS03.npz \
  --model_id ${data_name}_${seq_len}_${p} \
  --model ${model_name} \
  --data ${data_name} \
  --features M \
  --seq_len ${seq_len} \
  --label_len 48 \
  --pred_len ${p} \
  --enc_in 358 \
  --des Exp \
  --batch_size 16 \
  --alpha ${alpha} \
  --learning_rate 0.00045 \
  --train_epochs 30 \
  --lradj type3 \
  --patience 3 \
  --fusion_layers 1 \
  --d_model 256 \
  --top_k 3 \
  --num_kernel 3 \
  --encoder_layers 1 \
  --tfactor 1 \
  --revin 0 \
  --dropout 0 \
  --itr 1 \
  > logs/ShortForecasting/PCFNet/PEMS03_${alpha}_${model_name}_${p}.log
done

########################
# PEMS04
########################
for p in 12
do
python -u run.py \
  --is_training 1 \
  --root_path ${root}/PEMS/ \
  --data_path PEMS04.npz \
  --model_id ${data_name}_${seq_len}_${p} \
  --model ${model_name} \
  --data ${data_name} \
  --features M \
  --seq_len ${seq_len} \
  --label_len 48 \
  --pred_len ${p} \
  --enc_in 307 \
  --des Exp \
  --batch_size 16 \
  --alpha ${alpha} \
  --learning_rate 0.00045 \
  --train_epochs 30 \
  --lradj type3 \
  --patience 10 \
  --fusion_layers 1 \
  --d_model 256 \
  --top_k 3 \
  --num_kernel 3 \
  --encoder_layers 1 \
  --tfactor 1 \
  --revin 0 \
  --dropout 0 \
  --itr 1 \
  > logs/ShortForecasting/PCFNet/PEMS04_${alpha}_${model_name}_${p}.log
done

########################
# PEMS07
########################
for p in 12
do
python -u run.py \
  --is_training 1 \
  --root_path ${root}/PEMS/ \
  --data_path PEMS07.npz \
  --model_id ${data_name}_${seq_len}_${p} \
  --model ${model_name} \
  --data ${data_name} \
  --features M \
  --seq_len ${seq_len} \
  --label_len 48 \
  --pred_len ${p} \
  --enc_in 883 \
  --des Exp \
  --batch_size 4 \
  --alpha ${alpha} \
  --learning_rate 0.0004 \
  --train_epochs 30 \
  --lradj type3 \
  --patience 10 \
  --fusion_layers 1 \
  --d_model 256 \
  --top_k 3 \
  --num_kernel 3 \
  --encoder_layers 1 \
  --tfactor 1 \
  --revin 0 \
  --dropout 0 \
  --itr 1 \
  > logs/ShortForecasting/PCFNet/PEMS07_${alpha}_${model_name}_${p}.log
done

########################
# PEMS08
########################
for p in 12
do
python -u run.py \
  --is_training 1 \
  --root_path ${root}/PEMS/ \
  --data_path PEMS08.npz \
  --model_id ${data_name}_${seq_len}_${p} \
  --model ${model_name} \
  --data ${data_name} \
  --features M \
  --seq_len ${seq_len} \
  --label_len 48 \
  --pred_len ${p} \
  --enc_in 170 \
  --des Exp \
  --batch_size 16 \
  --alpha ${alpha} \
  --learning_rate 0.0008 \
  --train_epochs 30 \
  --lradj type3 \
  --patience 10 \
  --fusion_layers 1 \
  --d_model 256 \
  --top_k 1 \
  --num_kernel 3 \
  --encoder_layers 1 \
  --tfactor 1 \
  --revin 1 \
  --dropout 0 \
  --itr 1 \
  > logs/ShortForecasting/PCFNet/PEMS08_${alpha}_${model_name}_${p}.log
done

echo "All experiments completed!"
