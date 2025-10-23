@echo off
setlocal enabledelayedexpansion

if not exist ".\logs" mkdir .\logs
if not exist ".\logs\LongForecasting" mkdir .\logs\LongForecasting
if not exist ".\logs\LongForecasting\PCFNet" mkdir .\logs\LongForecasting\PCFNet

set model_name=PCFNet
set seq_len=96
set GPU=0
set root=./dataset

REM ETTh1 experiments
set alpha=0.35
set data_name=ETTh1
for %%p in (96 192 336 720) do (
  set CUDA_VISIBLE_DEVICES=!GPU!
  python -u run.py ^
    --is_training 1 ^
    --root_path !root!/ETT-small/ ^
    --data_path !data_name!.csv ^
    --model_id !data_name!_!seq_len!_%%p ^
    --model !model_name! ^
    --data !data_name! ^
    --features M ^
    --seq_len !seq_len! ^
    --label_len 48 ^
    --pred_len %%p ^
    --enc_in 7 ^
    --des Exp ^
    --batch_size 64 ^
    --alpha !alpha! ^
    --learning_rate 0.0004 ^
    --train_epochs 10 ^
    --patience 3 ^
    --fusion_layers 1 ^
    --d_model 64 ^
    --top_k 3 ^
    --num_kernel 3 ^
    --encoder_layers 2 ^
    --tfactor 0.5 ^
    --itr 1 > logs\LongForecasting\PCFNet\!data_name!_!alpha!_!model_name!_%%p.logs
)

REM ETTh2 experiments
set alpha=0.35
set data_name=ETTh2
for %%p in (96 192 336 720) do (
  set CUDA_VISIBLE_DEVICES=!GPU!
  python -u run.py ^
    --is_training 1 ^
    --root_path !root!/ETT-small/ ^
    --data_path !data_name!.csv ^
    --model_id !data_name!_!seq_len!_%%p ^
    --model !model_name! ^
    --data !data_name! ^
    --features M ^
    --seq_len !seq_len! ^
    --label_len 48 ^
    --pred_len %%p ^
    --enc_in 7 ^
    --des Exp ^
    --batch_size 64 ^
    --alpha !alpha! ^
    --learning_rate 0.0002 ^
    --train_epochs 10 ^
    --patience 3 ^
    --fusion_layers 2 ^
    --d_model 256 ^
    --top_k 3 ^
    --num_kernel 5 ^
    --encoder_layers 2 ^
    --tfactor 0.5 ^
    --itr 1 > logs\LongForecasting\PCFNet\!data_name!_!alpha!_!model_name!_%%p.logs
)

REM ETTm1 experiments
set alpha=0.35
set data_name=ETTm1
for %%p in (96 192 336 720) do (
  set CUDA_VISIBLE_DEVICES=!GPU!
  python -u run.py ^
    --is_training 1 ^
    --root_path !root!/ETT-small/ ^
    --data_path !data_name!.csv ^
    --model_id !data_name!_!seq_len!_%%p ^
    --model !model_name! ^
    --data !data_name! ^
    --features M ^
    --seq_len !seq_len! ^
    --label_len 48 ^
    --pred_len %%p ^
    --enc_in 7 ^
    --des Exp ^
    --batch_size 64 ^
    --alpha !alpha! ^
    --learning_rate 0.00015 ^
    --train_epochs 10 ^
    --patience 3 ^
    --fusion_layers 1 ^
    --d_model 256 ^
    --top_k 2 ^
    --num_kernel 5 ^
    --encoder_layers 2 ^
    --tfactor 1 ^
    --itr 1 > logs\LongForecasting\PCFNet\!data_name!_!alpha!_!model_name!_%%p.logs
)

REM ETTm2 experiments
set alpha=0.35
set data_name=ETTm2
for %%p in (96 192 336 720) do (
  set CUDA_VISIBLE_DEVICES=!GPU!
  python -u run.py ^
    --is_training 1 ^
    --root_path !root!/ETT-small/ ^
    --data_path !data_name!.csv ^
    --model_id !data_name!_!seq_len!_%%p ^
    --model !model_name! ^
    --data !data_name! ^
    --features M ^
    --seq_len !seq_len! ^
    --label_len 48 ^
    --pred_len %%p ^
    --enc_in 7 ^
    --des Exp ^
    --batch_size 64 ^
    --alpha !alpha! ^
    --learning_rate 0.00015 ^
    --train_epochs 10 ^
    --patience 3 ^
    --fusion_layers 1 ^
    --d_model 128 ^
    --top_k 3 ^
    --num_kernel 3 ^
    --encoder_layers 2 ^
    --tfactor 0.5 ^
    --itr 1 > logs\LongForecasting\PCFNet\!data_name!_!alpha!_!model_name!_%%p.logs
)

REM Weather experiments
set alpha=0.1
set data_name=weather
for %%p in (96 192 336 720) do (
  set CUDA_VISIBLE_DEVICES=!GPU!
  python -u run.py ^
    --is_training 1 ^
    --root_path !root!/weather/ ^
    --data_path weather.csv ^
    --model_id !data_name!_!seq_len!_%%p ^
    --model !model_name! ^
    --data custom ^
    --features M ^
    --seq_len !seq_len! ^
    --label_len 48 ^
    --pred_len %%p ^
    --enc_in 21 ^
    --des Exp ^
    --batch_size 32 ^
    --alpha !alpha! ^
    --learning_rate 0.00025 ^
    --train_epochs 20 ^
    --patience 3 ^
    --fusion_layers 1 ^
    --d_model 256 ^
    --top_k 3 ^
    --num_kernel 5 ^
    --encoder_layers 3 ^
    --tfactor 2 ^
    --itr 1 > logs\LongForecasting\PCFNet\!data_name!_!alpha!_!model_name!_%%p.logs
)

REM Solar experiments
set alpha=0.05
set data_name=Solar
for %%p in (96 192 336 720) do (
  set CUDA_VISIBLE_DEVICES=!GPU!
  python -u run.py ^
    --is_training 1 ^
    --root_path !root!/Solar/ ^
    --data_path solar_AL.txt ^
    --model_id !data_name!_!seq_len!_%%p ^
    --model !model_name! ^
    --data Solar ^
    --features M ^
    --seq_len !seq_len! ^
    --label_len 48 ^
    --pred_len %%p ^
    --enc_in 137 ^
    --des Exp ^
    --batch_size 32 ^
    --alpha !alpha! ^
    --learning_rate 0.0011 ^
    --train_epochs 15 ^
    --patience 3 ^
    --fusion_layers 1 ^
    --d_model 256 ^
    --top_k 1 ^
    --num_kernel 3 ^
    --encoder_layers 1 ^
    --tfactor 1 ^
    --itr 1 > logs\LongForecasting\PCFNet\!data_name!_!alpha!_!model_name!_%%p.logs
)

REM Electricity experiments
set alpha=0.2
set data_name=electricity
for %%p in (96 192 336 720) do (
  set CUDA_VISIBLE_DEVICES=!GPU!
  python -u run.py ^
    --is_training 1 ^
    --root_path !root!/electricity/ ^
    --data_path electricity.csv ^
    --model_id !data_name!_!seq_len!_%%p ^
    --model !model_name! ^
    --data custom ^
    --features M ^
    --seq_len !seq_len! ^
    --label_len 48 ^
    --pred_len %%p ^
    --enc_in 321 ^
    --des Exp ^
    --batch_size 8 ^
    --alpha !alpha! ^
    --learning_rate 0.0006 ^
    --train_epochs 20 ^
    --patience 3 ^
    --fusion_layers 1 ^
    --d_model 512 ^
    --top_k 3 ^
    --num_kernel 3 ^
    --encoder_layers 1 ^
    --tfactor 1 ^
    --itr 1 > logs\LongForecasting\PCFNet\!data_name!_!alpha!_!model_name!_%%p.logs
)

REM Traffic experiments
set alpha=0.35
set data_name=traffic
for %%p in (96 192 336 720) do (
  set CUDA_VISIBLE_DEVICES=!GPU!
  python -u run.py ^
    --is_training 1 ^
    --root_path !root!/traffic/ ^
    --data_path traffic.csv ^
    --model_id !data_name!_!seq_len!_%%p ^
    --model !model_name! ^
    --data custom ^
    --features M ^
    --seq_len !seq_len! ^
    --label_len 48 ^
    --pred_len %%p ^
    --enc_in 862 ^
    --des Exp ^
    --batch_size 2 ^
    --alpha !alpha! ^
    --learning_rate 0.0008 ^
    --train_epochs 30 ^
    --patience 3 ^
    --fusion_layers 1 ^
    --d_model 512 ^
    --top_k 3 ^
    --num_kernel 3 ^
    --encoder_layers 3 ^
    --tfactor 1 ^
    --lradj TST ^
    --itr 1 > logs\LongForecasting\PCFNet\!data_name!_!alpha!_!model_name!_%%p.logs
)

echo All experiments completed!
endlocal