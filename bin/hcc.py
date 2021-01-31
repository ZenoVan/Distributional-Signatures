import os

datasets = ["20newsgroup", "amazon", "huffpost", "reuters"]
data_paths = ["../data/20news.json", "../data/amazon.json", "../data/huffpost.json", "../data/reuters.json"]
n_train_classes = ["8", "10", "20", "15"]
n_val_classes = ["5", "5", "5", "5"]
n_test_classes = ["7", "9", "16", "11"]

for seed in ["3", "15", "172"]:
    for shot in ["5", "1"]:
        for i in range(4):
            cmd = "nohup python ../src/main_gan.py \
                --cuda 0 \
                --way 5 \
                --shot "+shot+" \
                --query 25 \
                --mode train \
                --embedding mlad \
                --classifier r2d2 \
                --dataset=" + datasets[i] + " \
                --data_path=" + data_paths[i] + " \
                --n_train_class=" + n_train_classes[i] + " \
                --n_val_class=" + n_val_classes[i] + " \
                --n_test_class=" + n_test_classes[i] + " \
                --train_episodes 100 \
                --k 1 \
                --lr_g 1e-3 \
                --lr_d 1e-3 \
                --Comments 'a: " + str(i) + "' \
                --patience 20 \
                --seed "+seed+" \
                >out/" + datasets[i] + ".shot" + shot + ".seed" + seed
            os.system(cmd)





