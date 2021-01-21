#dataset=amazon
#data_path="../data/amazon.json"
#n_train_class=10
#n_val_class=5
#n_test_class=9

#dataset=fewrel
#data_path="../data/fewrel.json"
#n_train_class=65
#n_val_class=5
#n_test_class=10
#
dataset=20newsgroup
data_path="../data/20news.json"
n_train_class=8
n_val_class=5
n_test_class=7
#
#dataset=huffpost
#data_path="../data/huffpost.json"
#n_train_class=20
#n_val_class=5
#n_test_class=16
#
#dataset=rcv1
#data_path="../data/rcv1.json"
#n_train_class=37
#n_val_class=10
#n_test_class=24
#
#dataset=reuters
#data_path="../data/reuters.json"
#n_train_class=15
#n_val_class=5
#n_test_class=11

# if [ "$dataset" = "fewrel" ]; then
#    python ../src/main_gan.py \
#        --cuda 0 \
#        --way 5 \
#        --shot 5 \
#        --query 25 \
#        --mode train \
#        --embedding mlad \
#        --classifier r2d2 \
#        --dataset=$dataset \
#        --data_path=$data_path \
#        --n_train_class=$n_train_class \
#        --n_val_class=$n_val_class \
#        --n_test_class=$n_test_class \
##        --auxiliary pos \
##        --meta_iwf \
##        --meta_w_target \
##        --pretrain="../bin/tmp-runs-pretrain/16082801702344212/99.pretrain_ebd"
#else
python ../src/main_gan.py \
    --cuda 0 \
    --way 5 \
    --shot 1 \
    --query 25 \
    --mode train \
    --embedding mlad \
    --classifier r2d2 \
    --dataset=$dataset \
    --data_path=$data_path \
    --n_train_class=$n_train_class \
    --n_val_class=$n_val_class \
    --n_test_class=$n_test_class \
    --train_episodes 100 \
    --k 1 \
    --lr_g 1e-3 \
    --lr_d 1e-3 \
    --Comments "debug" \
    --patience 20 \
#    --ablation "-IL"
        # --train_mode t_add_v
        # --lr_scheduler ExponentialLR\
        # --ExponentialLR_gamma 0.98
#        --pretrain="../bin/tmp-runs-pretrain/16082802589535282/1.pretrain_ebd"
#        --wv_path = 'pretrain_wordvec' \
#        --word_vector = 'pretrain_wordvec/wiki.en.vec'
#        --meta_iwf \
#        --meta_w_target
fi
