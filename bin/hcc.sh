dataset=huffpost
# data_path="../data/huffpost.json"
n_train_class=20
n_val_class=5
n_test_class=16

datasets=("huffpost" "amazon" "20news")
for a in {0..2}
do
    nohup python ../src/main_gan.py \
        --cuda 0 \
        --way 5 \
        --shot 5 \
        --query 25 \
        --mode train \
        --embedding mlad \
        --classifier r2d2 \
        --dataset=${datasets[$a]} \
        --data_path=../data/${datasets[$a]}.json \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --train_episodes 100 \
        --k 1 \
        --lr_g 1e-3 \
        --lr_d 1e-3 \
        --Comments comments$a \
        --patience 20 \
        --seed 3 \
        > out/out$a
done