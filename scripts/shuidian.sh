export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

pred_len=1
state='hejiang'
onlytest_state='hejiang'
features='S'
target_name='maximum_load'
seq_length=10
training_time=0

for state in 'puge' 'shuidianjituan' 'hejiang' 'pingwu' 'kaijiang' 'meigu' 'santai' 'wanyuan' 'zhaojue' 
do
for target_name in 'energy_consumption' #'power_supply' 'power_generation' #'maximum_load'
do
python main.py \
    --data_name $state \
    --features $features \
    --seq_length $seq_length \
    --pred_len $pred_len\
    --input_size 1 \
    --batch_size 2 \
    --training_time $training_time \
    --target_name $target_name \
    --onlytest 0 \
    --final_pred_len 160\
    --is_recursion false\
    --is_store false\
    --test_load_path './trained_models_shuidian/'$state'_'$target_name'_1.pth'
done
done


