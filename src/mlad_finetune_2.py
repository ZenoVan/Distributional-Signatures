# 这个预训练是不包含 类sample 的，有多少类咱们就做多少类的文本分类任务

import os
import sys
import time
import datetime
import pickle
import signal
import argparse
import traceback

import numpy as np

import embedding.factory as ebd
import classifier.factory as clf
import dataset.loader as loader
import train.factory as train_utils
from classifier.base import BASE as base

from tqdm import tqdm
from termcolor import colored

from dataset.parallel_sampler import ParallelSampler
from train.utils import named_grad_param, grad_param, get_norm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from embedding.meta import RNN
from embedding.wordebd import WORDEBD
from embedding.GRL import ReverseLayerF


class MLAD_FTN(nn.Module):

    def __init__(self, ebd, args):
        super(MLAD_FTN, self).__init__()

        self.args = args

        self.ebd = ebd
        # self.aux = get_embedding(args)

        self.ebd_dim = self.ebd.embedding_dim

        self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=1, batch_first=True, dropout=0)
        self.rnn = RNN(300, 128, 1, True, 0)

        self.seq = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 1),
                    )

        self.d = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(500, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, args.n_train_class + args.n_val_class),
                )

    def forward(self, data, return_score=False):

        # 将单词转为词向量
        # print("\ndata.shape:", data['text'].shape)
        ebd = self.ebd(data)
        # w2v = ebd

        # scale = self.compute_score(data, ebd)
        # print("\ndata.shape:", ebd.shape)  # [b, text_len, 300]

        # Generator部分
        # print('_________________________________________ebd:', ebd.shape)
        # ebd, (hn, cn) = self.lstm(ebd)
        ebd = self.rnn(ebd, data['text_len'])
        # print("\n_______________________________________________ebd.shape:", ebd.shape)  # [b, text_len, 256]
        # for i, b in enumerate(ebd):
        #     ebd[i] = self.seq(ebd[i])
        # ebd = ebd[:, -1, :].reshape((-1, 256))
        # ebd = torch.max(ebd, dim=-1, keepdim=False)[0]
        ebd = self.seq(ebd).squeeze(-1)  # [b, text_len, 256] -> [b,text_len]
        # print("\ndata.shape:", ebd.shape)  # [b, text_len]
        word_weight = F.softmax(ebd, dim=-1)
        # print("word_weight.shape:", word_weight.shape)  # [b, text_len]
        # sentence_ebd = torch.sum((torch.unsqueeze(word_weight, dim=-1)) * w2v, dim=-2)
        # print("sentence_ebd.shape:", sentence_ebd.shape)

        reverse_feature = ReverseLayerF.apply(word_weight, 0.5) # 用梯度反向层
        # reverse_feature = word_weight  # 不用梯度反向层

        # 将reverse_feature统一变为[b, 500]，长则截断，短则补0
        if reverse_feature.shape[1] < 500:
            zero = torch.zeros((reverse_feature.shape[0], 500 - reverse_feature.shape[1]))
            if self.args.cuda != -1:
                zero = zero.cuda(self.args.cuda)
            reverse_feature = torch.cat((reverse_feature, zero), dim=-1)
            # print('reverse_feature.shape[1]', reverse_feature.shape[1])
        else:
            reverse_feature = reverse_feature[:, :500]
            # print('reverse_feature.shape[1]', reverse_feature.shape[1])

        # 通过判别器
        logits = self.d(reverse_feature)  # [b, 500] -> [b, args.n_train_class]

        return logits


def parse_args():

    parser = argparse.ArgumentParser(
            description="Meta-Learning Adaption Domain.")

    # data configuration
    parser.add_argument("--data_path", type=str,
                        default="data/reuters.json",
                        help="path to dataset")
    parser.add_argument("--dataset", type=str, default="reuters",
                        help="name of the dataset. "
                        "Options: [20newsgroup, amazon, huffpost, "
                        "reuters, fewrel]")
    parser.add_argument("--n_train_class", type=int, default=15,
                        help="number of meta-train classes")
    parser.add_argument("--n_val_class", type=int, default=5,
                        help="number of meta-val classes")
    parser.add_argument("--n_test_class", type=int, default=11,
                        help="number of meta-test classes")

    # load bert embeddings for sent-level datasets (optional)
    parser.add_argument("--n_workers", type=int, default=10,
                        help="Num. of cores used for loading data. Set this "
                        "to zero if you want to use all the cpus.")
    parser.add_argument("--bert", default=False, action="store_true",
                        help=("set true if use bert embeddings "
                              "(only available for sent-level datasets: "
                              "huffpost, fewrel"))
    parser.add_argument("--bert_cache_dir", default=None, type=str,
                        help=("path to the cache_dir of transformers"))
    parser.add_argument("--pretrained_bert", default=None, type=str,
                        help=("path to the pre-trained bert embeddings."))

    # task configuration
    parser.add_argument("--way", type=int, default=5,
                        help="#classes for each task")
    parser.add_argument("--shot", type=int, default=5,
                        help="#support examples for each class for each task")
    parser.add_argument("--query", type=int, default=25,
                        help="#query examples for each class for each task")

    # train/test configuration
    parser.add_argument("--train_epochs", type=int, default=1000,
                        help="max num of training epochs")
    parser.add_argument("--train_episodes", type=int, default=100,
                        help="#tasks sampled during each training epoch")
    parser.add_argument("--val_episodes", type=int, default=100,
                        help="#asks sampled during each validation epoch")
    parser.add_argument("--test_episodes", type=int, default=1000,
                        help="#tasks sampled during each testing epoch")

    # settings for finetuning baseline
    parser.add_argument("--finetune_loss_type", type=str, default="softmax",
                        help="type of loss for finetune top layer"
                        "options: [softmax, dist]")
    parser.add_argument("--finetune_maxepochs", type=int, default=5000,
                        help="number epochs to finetune each task for (inner loop)")
    parser.add_argument("--finetune_episodes", type=int, default=10,
                        help="number tasks to finetune for (outer loop)")
    parser.add_argument("--finetune_split", default=0.8, type=float,
                        help="percent of train data to allocate for val"
                             "when mode is finetune")

    # model options
    parser.add_argument("--embedding", type=str, default="avg",
                        help=("document embedding method. Options: "
                              "[avg, tfidf, meta, oracle, cnn]"))
    parser.add_argument("--classifier", type=str, default="nn",
                        help=("classifier. Options: [nn, proto, r2d2, mlp]"))
    parser.add_argument("--auxiliary", type=str, nargs="*", default=[],
                        help=("auxiliary embeddings (used for fewrel). "
                              "Options: [pos, ent]"))

    # cnn configuration
    parser.add_argument("--cnn_num_filters", type=int, default=50,
                        help="Num of filters per filter size [default: 50]")
    parser.add_argument("--cnn_filter_sizes", type=int, nargs="+",
                        default=[3, 4, 5],
                        help="Filter sizes [default: 3]")

    # nn configuration
    parser.add_argument("--nn_distance", type=str, default="l2",
                        help=("distance for nearest neighbour. "
                              "Options: l2, cos [default: l2]"))

    # proto configuration
    parser.add_argument("--proto_hidden", nargs="+", type=int,
                        default=[300, 300],
                        help=("hidden dimension of the proto-net"))

    # maml configuration
    parser.add_argument("--maml", action="store_true", default=False,
                        help=("Use maml or not. "
                        "Note: maml has to be used with classifier=mlp"))
    parser.add_argument("--mlp_hidden", nargs="+", type=int, default=[300, 5],
                        help=("hidden dimension of the proto-net"))
    parser.add_argument("--maml_innersteps", type=int, default=10)
    parser.add_argument("--maml_batchsize", type=int, default=10)
    parser.add_argument("--maml_stepsize", type=float, default=1e-1)
    parser.add_argument("--maml_firstorder", action="store_true", default=False,
                        help="truncate higher order gradient")

    # lrd2 configuration
    parser.add_argument("--lrd2_num_iters", type=int, default=5,
                        help=("num of Newton steps for LRD2"))

    # induction networks configuration
    parser.add_argument("--induct_rnn_dim", type=int, default=128,
                        help=("Uni LSTM dim of induction network's encoder"))
    parser.add_argument("--induct_hidden_dim", type=int, default=100,
                        help=("tensor layer dim of induction network's relation"))
    parser.add_argument("--induct_iter", type=int, default=3,
                        help=("num of routings"))
    # parser.add_argument("--induct_att_dim", type=int, default=64,
    #                     help=("attention projection dim of induction network"))

    # aux ebd configuration (for fewrel)
    parser.add_argument("--pos_ebd_dim", type=int, default=5,
                        help="Size of position embedding")
    parser.add_argument("--pos_max_len", type=int, default=40,
                        help="Maximum sentence length for position embedding")

    # base word embedding
    parser.add_argument("--wv_path", type=str,
                        default='../pretrain_wordvec',
                        help="path to word vector cache")
    parser.add_argument("--word_vector", type=str, default='../pretrain_wordvec/wiki.en.vec',
                        help=("Name of pretrained word embeddings."))
    parser.add_argument("--finetune_ebd", action="store_true", default=False,
                        help=("Finetune embedding during meta-training"))

    # options for the distributional signatures
    parser.add_argument("--meta_idf", action="store_true", default=False,
                        help="use idf")
    parser.add_argument("--meta_iwf", action="store_true", default=False,
                        help="use iwf")
    parser.add_argument("--meta_w_target", action="store_true", default=False,
                        help="use target importance score")
    parser.add_argument("--meta_w_target_lam", type=float, default=1,
                        help="lambda for computing w_target")
    parser.add_argument("--meta_target_entropy", action="store_true", default=False,
                        help="use inverse entropy to model task-specific importance")
    parser.add_argument("--meta_ebd", action="store_true", default=False,
                        help="use word embedding into the meta model "
                        "(showing that revealing word identity harm performance)")

    # training options
    parser.add_argument("--seed", type=int, default=330, help="seed")
    parser.add_argument("--dropout", type=float, default=0.1, help="drop rate")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--patience", type=int, default=20, help="patience")
    parser.add_argument("--clip_grad", type=float, default=None,
                        help="gradient clipping")
    parser.add_argument("--cuda", type=int, default=-1,
                        help="cuda device, -1 for cpu")
    parser.add_argument("--mode", type=str, default="test",
                        help=("Running mode."
                              "Options: [train, test, finetune]"
                              "[Default: test]"))
    parser.add_argument("--save", action="store_true", default=False,
                        help="train the model")
    parser.add_argument("--notqdm", action="store_true", default=False,
                        help="disable tqdm")
    parser.add_argument("--result_path", type=str, default="")
    parser.add_argument("--snapshot", type=str, default="",
                        help="path to the pretraiend weights")

    return parser.parse_args()


def print_args(args):
    """
        Print arguments (only show the relevant arguments)
    """
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        if args.embedding != "cnn" and attr[:4] == "cnn_":
            continue
        if args.classifier != "proto" and attr[:6] == "proto_":
            continue
        if args.classifier != "nn" and attr[:3] == "nn_":
            continue
        if args.embedding != "meta" and attr[:5] == "meta_":
            continue
        if args.embedding != "cnn" and attr[:4] == "cnn_":
            continue
        if args.classifier != "mlp" and attr[:4] == "mlp_":
            continue
        if args.classifier != "proto" and attr[:6] == "proto_":
            continue
        if "pos" not in args.auxiliary and attr[:4] == "pos_":
            continue
        if not args.maml and attr[:5] == "maml_":
            continue
        print("\t{}={}".format(attr.upper(), value))
    print("""
    ◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇
    ◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇
    ◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇
    ◇◇◇◆◆◆◆◆◇◇◆◆◆◆◆◇◇◇◇◇◇◇◇◇◇◆◆◆◆◆◇◆◇◇◇◇◇◇◇◇◇◇◇◆◆◆◆◆◇◆◇◇◇◇
    ◇◇◇◆◆◆◆◆◇◇◆◆◆◆◇◇◇◇◇◇◇◇◇◇◆◆◆◆◆◆◆◆◇◇◇◇◇◇◇◇◇◇◆◆◆◆◆◆◆◆◇◇◇◇
    ◇◇◇◇◇◆◆◇◇◇◇◆◆◇◇◇◇◇◇◇◇◇◇◆◆◆◇◇◇◇◆◆◇◇◇◇◇◇◇◇◇◆◆◆◇◇◇◇◆◆◇◇◇◇
    ◇◇◇◇◇◆◆◇◇◇◇◆◆◇◇◇◇◇◇◇◇◇◇◆◆◇◇◇◇◇◇◆◇◇◇◇◇◇◇◇◇◆◆◇◇◇◇◇◇◆◇◇◇◇
    ◇◇◇◇◇◆◆◇◇◇◇◆◆◇◇◇◇◇◇◇◇◇◇◆◆◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◆◆◇◇◇◇◇◇◇◇◇◇◇
    ◇◇◇◇◇◆◆◆◆◆◆◆◆◇◇◇◇◇◇◇◇◇◆◆◆◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◆◆◆◇◇◇◇◇◇◇◇◇◇◇
    ◇◇◇◇◇◆◆◇◇◇◇◆◆◇◇◇◇◇◇◇◇◇◆◆◆◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◆◆◆◇◇◇◇◇◇◇◇◇◇◇
    ◇◇◇◇◇◆◆◇◇◇◇◆◆◇◇◇◇◇◇◇◇◇◇◆◆◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◆◆◇◇◇◇◇◇◇◇◇◇◇
    ◇◇◇◇◇◆◆◇◇◇◇◆◆◇◇◇◇◇◇◇◇◇◇◆◆◇◇◇◇◇◇◆◆◇◇◇◇◇◇◇◇◆◆◇◇◇◇◇◇◆◆◇◇◇
    ◇◇◇◇◆◆◆◇◇◇◇◆◆◆◇◇◇◇◇◇◇◇◇◆◆◆◆◇◇◇◆◆◇◇◇◇◇◇◇◇◇◆◆◆◆◇◇◇◆◆◇◇◇◇
    ◇◇◇◆◆◆◆◆◇◇◆◆◆◆◆◇◇◇◇◇◇◇◇◇◆◆◆◆◆◆◆◇◇◇◇◇◇◇◇◇◇◇◆◆◆◆◆◆◆◇◇◇◇◇
    ◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◆◆◆◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◆◆◆◇◇◇◇◇◇◇
    ◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇
    ◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇
    ◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇
    """)


def set_seed(seed):
    """
        Setting random seeds
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def reidx_y(YS, args):
    '''
        Map the labels into 0,..., way
        @param YS: batch_size

        @return YS_new: batch_size
    '''
    unique1, inv_S = torch.unique(YS, sorted=True, return_inverse=True)

    Y_new = torch.arange(start=0, end=args.n_train_class, dtype=unique1.dtype,
                         device=unique1.device)

    return Y_new[inv_S]


def train(train_data, model, args):
    '''
        Train the model
    '''
    # creating a tmp directory to save the models
    out_dir = os.path.abspath(os.path.join(
        os.path.curdir,
        "tmp-runs-pretrain",
        str(int(time.time() * 1e7))))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # best_path = None

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, 'max', patience=args.patience // 2, factor=0.1, verbose=True)

    print("{}, Start training".format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')), flush=True)

    # 把label变为0-9（如果是10类的话）
    key = np.sort(np.unique(train_data['label']))
    print("\n-------------------------------------------------------------------------------------")
    print(key)
    temp_list = []
    for i in train_data['label']:
        temp_list.append(list(key).index(i))
    temp_list = np.array(temp_list)
    train_data['label'] = temp_list
    print(train_data['label'].shape)
    print(set(train_data['label']))
    print("\n-------------------------------------------------------------------------------------")

    train_data_batch = DataLoader(Data_Class(train_data), batch_size=32, shuffle=True)

    for ep in range(100):

        model.train()

        for text, label, text_len in train_data_batch:

            # print("train_data_batch:", task.shape)
            data = {}
            if args.cuda != -1:
                data['text'] = text.cuda(args.cuda)
                data['text_len'] = text_len.cuda(args.cuda)
                label = label.cuda(args.cuda)
            else:
                data['text'] = text
                data['text_len'] = text_len
            # print(data['text'].shape, data['text_len'].shape)
            pred = model(data)
            # print("pred:", pred)
            # print("\n___________________________________reixy_label:", label.tolist())
            # print(pred.shape, label.shape)
            loss = F.cross_entropy(pred, label)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if ep % 10 == 0:

            acc, std = test(train_data_batch, model, args)
            print("{}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f} ".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                "ep", ep,
                colored("train", "red"),
                colored("acc:", "blue"), acc, std,
            ), flush=True)

        save_path = os.path.join(out_dir, str(ep))
        print("{}, Save cur {}th model to {}".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'), ep,
            save_path))
        torch.save(model.state_dict(), save_path + '.pretrain_ebd')

    print("{}, End of training. Restore the weights".format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')),
        flush=True)

    # restore the best saved model
    # model['ebd'].load_state_dict(torch.load(best_path + '.pretrain_ebd'))
    # model['clf'].load_state_dict(torch.load(best_path + '.clf'))

    # if args.save:
    #     # save the current model
    #     out_dir = os.path.abspath(os.path.join(
    #         os.path.curdir,
    #         "saved-runs",
    #         str(int(time.time() * 1e7))))
    #     if not os.path.exists(out_dir):
    #         os.makedirs(out_dir)
    #
    #     best_path = os.path.join(out_dir, 'best')
    #
    #     print("{}, Save best model to {}".format(
    #         datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
    #         best_path), flush=True)
    #
    #     torch.save(model['ebd'].state_dict(), best_path + '.pretrain_ebd')
    #     # torch.save(model['clf'].state_dict(), best_path + '.clf')
    #
    #     with open(best_path + '_args.txt', 'w') as f:
    #         for attr, value in sorted(args.__dict__.items()):
    #             f.write("{}={}\n".format(attr, value))

    return


def test(train_data_batch, model, args):
    '''
        Evaluate the model on a bag of sampled tasks. Return the mean accuracy
        and its std.
    '''
    model.eval()

    acc = []

    for task, label, text_len in train_data_batch:

        acc.append(test_one(task, label, text_len, model, args))

    acc = np.array(acc)

    return np.mean(acc), np.std(acc)


def test_one(task, label, text_len, model, args):
    '''
        Evaluate the model on one sampled task. Return the accuracy.
    '''

    data = {}
    if args.cuda != -1:
        data['text'] = task.cuda(args.cuda)
        data['text_len'] = text_len.cuda(args.cuda)
        label = label.cuda(args.cuda)
    else:
        data['text'] = task
        data['text_len'] = text_len
    pred = model(data)
    # print("pred:", pred)
    # print("\n___________________________________reixy_label:", label.tolist())
    logits = F.softmax(pred)
    # print("------------------------------logits,label___", logits.shape, label.shape)
    # print("label:", label)
    acc = compute_acc(logits, label)

    return acc


def compute_acc(pred, true):
    '''
        Compute the accuracy.
        @param pred: batch_size * num_classes
        @param true: batch_size
    '''
    return torch.mean((torch.argmax(pred, dim=1) == true).float()).item()


class Data_Class(Dataset):

    def __init__(self, train_data):
        self.data = train_data['text']
        self.label = train_data['label']
        self.text_len = train_data['text_len']

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        text_len = self.text_len[index]

        return data, label, text_len

    def __len__(self):
        return len(self.label)



def main():

    args = parse_args()

    print_args(args)

    set_seed(args.seed)

    # load data
    _train_data, _val_data, _test_data, vocab = loader.load_dataset(args)

    train_data = {}
    print(_train_data['text'].shape, _val_data['text'].shape)
    if _train_data['text'].shape[-1] > _val_data['text'].shape[-1]:
        _val_data['text'] = np.pad(_val_data['text'], ((0, 0), (0, _train_data['text'].shape[-1] - _val_data['text'].shape[-1])),
                                   'constant')
    elif _train_data['text'].shape[-1] < _val_data['text'].shape[-1]:
        _train_data['text'] = np.pad(_train_data['text'], ((0, 0), (0, _val_data['text'].shape[-1] - _train_data['text'].shape[-1])),
                                     'constant')

    print("\nafter:", _train_data['text'].shape, _val_data['text'].shape)
    train_data['text'] = np.concatenate((_train_data['text'], _val_data['text']), axis=0)
    train_data['label'] = np.concatenate((_train_data['label'], _val_data['label']), axis=0)
    train_data['text_len'] = np.concatenate((_train_data['text_len'], _val_data['text_len']), axis=0)

    set_label = list(set(train_data['label']))
    print("\n----------------set_label", set_label)

    # initialize model
    print("{}, Building embedding".format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')), flush=True)

    ebd = WORDEBD(vocab, args.finetune_ebd)

    model = MLAD_FTN(ebd, args)

    if args.cuda != -1:
        model = model.cuda(args.cuda)

    train(train_data, model, args)


if __name__ == '__main__':
    main()

