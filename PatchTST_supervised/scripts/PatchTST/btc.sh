if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/btc" ]; then
    mkdir ./logs/btc
fi
seq_len=240
model_name=PatchTST

root_path_name=../
data_path_name=BTCUSDT_1h_features.csv
model_id_name=btc
data_name=custom

random_seed=2021
for pred_len in 2
do
    python3 -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features MS \
      --target 'next_gain_is_big' \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --freq 'h' \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 300\
      --patience 15\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 --batch_size 1024 --learning_rate 0.0001 #>logs/btc/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
