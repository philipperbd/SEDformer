export CUDA_VISIBLE_DEVICES=1

#cd ..

for model in SEDformer
do

for preLen in 96 192 336 720
do

# weather
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/weather/ \
 --data_path weather.csv \
 --task_id weather \
 --model $model \
 --data custom \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --depth 3 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 21 \
 --dec_in 21 \
 --c_out 21 \
 --des 'Exp' \
 --itr 3
done


done

