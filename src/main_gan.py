import os
import sys
import time
import json
import datetime
import pickle
import signal
import argparse
import traceback

import dataset.utils as utils
from multiprocessing import Process, Queue, cpu_count

import torch
import numpy as np
from tqdm import tqdm
from termcolor import colored

import embedding.factory as ebd
import classifier.factory as clf
from classifier.base import BASE
import dataset.loader as loader
from dataset.utils import tprint
from embedding.meta import RNN
import train.factory as train_utils
from train.utils import named_grad_param, grad_param, get_norm
from embedding.wordebd import WORDEBD

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def parse_args():
    parser = argparse.ArgumentParser(
            description="Few Shot Text Classification with Distributional Signatures")

    # data configuration
    parser.add_argument("--data_path", type=str,
                        default="data/reuters.json",
                        help="path to dataset")
    parser.add_argument("--dataset", type=str, default="reuters",
                        help="name of the dataset. "
                        "Options: [20newsgroup, amazon, huffpost, "
                        "reuters, rcv1, fewrel]")
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

    # base word embedding
    parser.add_argument("--wv_path", type=str,
                        default='../pretrain_wordvec',
                        help="path to word vector cache")
    parser.add_argument("--word_vector", type=str, default='../pretrain_wordvec/wiki.en.vec',
                        help=("Name of pretrained word embeddings."))
    parser.add_argument("--finetune_ebd", action="store_true", default=False,
                        help=("Finetune embedding during meta-training"))

    # training options
    parser.add_argument("--seed", type=int, default=330, help="seed")
    parser.add_argument("--dropout", type=float, default=0.1, help="drop rate")
    parser.add_argument("--patience", type=int, default=20, help="patience")
    parser.add_argument("--clip_grad", type=float, default=None,
                        help="gradient clipping")
    parser.add_argument("--cuda", type=int, default=-1,
                        help="cuda device, -1 for cpu")
    parser.add_argument("--mode", type=str, default="test",
                        help=("Running mode."
                              "Options: [train, test]"
                              "[Default: test]"))
    parser.add_argument("--save", action="store_true", default=False,
                        help="train the model")
    parser.add_argument("--notqdm", action="store_true", default=False,
                        help="disable tqdm")
    parser.add_argument("--result_path", type=str, default="")
    parser.add_argument("--snapshot", type=str, default="",
                        help="path to the pretraiend weights")

    parser.add_argument("--pretrain", type=str, default=None, help="path to the pretraiend weights for MLAD")
    parser.add_argument("--k", type=int, default=None, help="Number of iterations of the adversarial network")
    parser.add_argument("--lr_g", type=float, default=1e-3, help="learning rate of G")
    parser.add_argument("--lr_d", type=float, default=1e-3, help="learning rate of D")
    parser.add_argument("--lr_scheduler", type=str, default=None, help="lr_scheduler")
    parser.add_argument("--ExponentialLR_gamma", type=float, default=0.98, help="ExponentialLR_gamma")
    parser.add_argument("--train_mode", type=str, default=None, help="you can choose t_add_v or None")
    parser.add_argument("--ablation", type=str, default="", help="ablation study:[-DAN, -IL]")
    parser.add_argument("--path_drawn_data", type=str, default="reuters_False_data.json", help="path_drawn_data")
    parser.add_argument("--Comments", type=str, default="", help="Comments")
    parser.add_argument("--id2word", default=None, help="id2word")

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


def get_embedding(vocab, args):
    print("{}, Building embedding".format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')), flush=True)

    ebd = WORDEBD(vocab, args.finetune_ebd)

    modelG = ModelG(ebd, args)
    modelD = ModelD(ebd, args)

    print("{}, Building embedding".format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')), flush=True)

    if args.cuda != -1:
        modelG = modelG.cuda(args.cuda)
        modelD = modelD.cuda(args.cuda)
        return modelG, modelD
    else:
        return modelG, modelD


def get_classifier(ebd_dim, args):
    tprint("Building classifier")

    model = R2D2(ebd_dim, args)

    if args.cuda != -1:
        return model.cuda(args.cuda)
    else:
        return model


def task_sampler(data, args):

    all_classes = np.unique(data['label'])
    num_classes = len(all_classes)

    # sample classes
    temp = np.random.permutation(num_classes)
    sampled_classes = temp[:args.way]

    source_classes = temp[args.way:args.way + args.way]

    return sampled_classes, source_classes  # 存的是idx_list的索引


class ModelG(nn.Module):

    def __init__(self, ebd, args):
        super(ModelG, self).__init__()

        self.args = args

        self.ebd = ebd
        # self.aux = get_embedding(args)

        self.ebd_dim = self.ebd.embedding_dim

        self.rnn = RNN(300, 128, 1, True, 0)
        self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=1, batch_first=True, dropout=0)

        self.seq = nn.Sequential(
            nn.Linear(256, 1),
        )

    def forward(self, data, flag=None, return_score=False):

        # 将单词转为词向量
        ebd = self.ebd(data)
        w2v = ebd

        avg_sentence_ebd = torch.mean(w2v, dim=1)

        # scale = self.compute_score(data, ebd)
        # print("\ndata.shape:", ebd.shape)  # [b, text_len, 300]

        # Generator部分
        ebd = self.rnn(ebd, data['text_len'])
        # ebd, (hn, cn) = self.lstm(ebd)
        # print("\ndata.shape:", ebd.shape)  # [b, text_len, 256]
        # for i, b in enumerate(ebd):
        ebd = self.seq(ebd).squeeze(-1)  # [b, text_len, 256] -> [b, text_len]
        # ebd = torch.max(ebd, dim=-1, keepdim=False)[0]
        # print("\ndata.shape:", ebd.shape)  # [b, text_len]
        word_weight = F.softmax(ebd, dim=-1)
        # print("word_weight.shape:", word_weight.shape)  # [b, text_len]
        sentence_ebd = torch.sum((torch.unsqueeze(word_weight, dim=-1)) * w2v, dim=-2)
        # print("sentence_ebd.shape:", sentence_ebd.shape)

        reverse_feature = word_weight

        # 将reverse_feature统一变为[b, 500]，长则截断，短则补0
        if reverse_feature.shape[1] < 500:
            zero = torch.zeros((reverse_feature.shape[0], 500-reverse_feature.shape[1]))
            if self.args.cuda != -1:
               zero = zero.cuda(self.args.cuda)
            reverse_feature = torch.cat((reverse_feature, zero), dim=-1)
            # print('reverse_feature.shape[1]', reverse_feature.shape[1])
        else:
            reverse_feature = reverse_feature[:, :500]
            # print('reverse_feature.shape[1]', reverse_feature.shape[1])

        if self.args.ablation == '-IL':
            sentence_ebd = torch.cat((avg_sentence_ebd, sentence_ebd), 1)
            print("%%%%%%%%%%%%%%%%%%%%This is ablation mode: -IL%%%%%%%%%%%%%%%%%%")

        return sentence_ebd, reverse_feature, avg_sentence_ebd


class ModelD(nn.Module):

    def __init__(self, ebd, args):
        super(ModelD, self).__init__()

        self.args = args

        self.ebd = ebd
        # self.aux = get_embedding(args)

        self.d = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(500, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 2),
                )

    def forward(self, reverse_feature):

        # 通过判别器
        logits = self.d(reverse_feature)  # [b, 500] -> [b, 2]

        return logits


class R2D2(BASE):
    '''
        META-LEARNING WITH DIFFERENTIABLE CLOSED-FORM SOLVERS
    '''
    def __init__(self, ebd_dim, args):
        super(R2D2, self).__init__(args)
        self.ebd_dim = ebd_dim

        self.args = args

        # meta parameters to learn
        self.lam = nn.Parameter(torch.tensor(-1, dtype=torch.float))
        self.alpha = nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.beta = nn.Parameter(torch.tensor(1, dtype=torch.float))
        # lambda and alpha is learned in the log space

        # cached tensor for speed
        self.I_support = nn.Parameter(
            torch.eye(self.args.shot * self.args.way, dtype=torch.float),
            requires_grad=False)
        self.I_way = nn.Parameter(torch.eye(self.args.way, dtype=torch.float),
                                  requires_grad=False)

    def _compute_w(self, XS, YS_onehot):
        '''
            Compute the W matrix of ridge regression
            @param XS: support_size x ebd_dim
            @param YS_onehot: support_size x way

            @return W: ebd_dim * way
        '''

        W = XS.t() @ torch.inverse(
                XS @ XS.t() + (10. ** self.lam) * self.I_support) @ YS_onehot

        return W

    def _label2onehot(self, Y):
        '''
            Map the labels into 0,..., way
            @param Y: batch_size

            @return Y_onehot: batch_size * ways
        '''
        Y_onehot = F.embedding(Y, self.I_way)

        return Y_onehot

    def forward(self, XS, YS, XQ, YQ, XQ_logitsD, XSource_logitsD, YQ_d, YSource_d, query_data=None):
        '''
            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size
            @param XQ (support x): query_size x ebd_dim
            @param YQ (support y): query_size

            @return acc
            @return loss
        '''

        YS, YQ = self.reidx_y(YS, YQ)

        YS_onehot = self._label2onehot(YS)

        W = self._compute_w(XS, YS_onehot)

        pred = (10.0 ** self.alpha) * XQ @ W + self.beta

        loss = F.cross_entropy(pred, YQ)

        acc = BASE.compute_acc(pred, YQ)

        d_acc = (BASE.compute_acc(XQ_logitsD, YQ_d) + BASE.compute_acc(XSource_logitsD, YSource_d)) / 2

        if query_data is not None:
            y_hat = torch.argmax(pred, dim=1)
            X_hat = query_data[y_hat != YQ]
            return acc, d_acc, loss, X_hat
        else:
            return acc, d_acc, loss, loss


class ParallelSampler():

    def __init__(self, data, args, sampled_classes, source_classes, num_episodes=None):
        self.data = data
        self.args = args
        self.num_episodes = num_episodes
        self.sampled_classes = sampled_classes
        self.source_classes = source_classes

        self.all_classes = np.unique(self.data['label'])
        self.num_classes = len(self.all_classes)
        if self.num_classes < self.args.way:
            raise ValueError("Total number of classes is less than #way.")

        self.idx_list = []
        for y in self.all_classes:
            self.idx_list.append(
                    np.squeeze(np.argwhere(self.data['label'] == y)))

        self.count = 0
        self.done_queue = Queue()

        self.num_cores = cpu_count() if args.n_workers is 0 else args.n_workers

        self.p_list = []
        for i in range(self.num_cores):
            self.p_list.append(
                    Process(target=self.worker, args=(self.done_queue, self.sampled_classes, self.source_classes)))

        for i in range(self.num_cores):
            self.p_list[i].start()

    def get_epoch(self):

        for _ in range(self.num_episodes):
            # wait until self.thread finishes
            support, query, source = self.done_queue.get()

            # convert to torch.tensor
            support = utils.to_tensor(support, self.args.cuda, ['raw'])
            query = utils.to_tensor(query, self.args.cuda, ['raw'])
            source = utils.to_tensor(source, self.args.cuda, ['raw'])

            support['is_support'] = True
            query['is_support'] = False
            source['is_support'] = False

            yield support, query, source

    def worker(self, done_queue, sampled_classes, source_classes):
        '''
            Generate one task (support and query).
            Store into self.support[self.cur] and self.query[self.cur]
        '''
        while True:
            if done_queue.qsize() > 100:
                time.sleep(1)
                continue

            # sample examples
            support_idx, query_idx, source_idx = [], [], []
            for y in sampled_classes:
                tmp = np.random.permutation(len(self.idx_list[y]))
                support_idx.append(
                        self.idx_list[y][tmp[:self.args.shot]])
                query_idx.append(
                        self.idx_list[y][
                            tmp[self.args.shot:self.args.shot+self.args.query]])

            for z in source_classes:
                tmp = np.random.permutation(len(self.idx_list[z]))
                source_idx.append(
                    tmp[:self.args.query]
                )

            support_idx = np.concatenate(support_idx)
            query_idx = np.concatenate(query_idx)
            source_idx = np.concatenate(source_idx)


            # aggregate examples
            max_support_len = np.max(self.data['text_len'][support_idx])
            max_query_len = np.max(self.data['text_len'][query_idx])
            max_source_len = np.max(self.data['text_len'][source_idx])

            support = utils.select_subset(self.data, {}, ['text', 'text_len', 'label'],
                                     support_idx, max_support_len)
            query = utils.select_subset(self.data, {}, ['text', 'text_len', 'label'],
                                   query_idx, max_query_len)
            source = utils.select_subset(self.data, {}, ['text', 'text_len', 'label'],
                                   source_idx, max_source_len)

            done_queue.put((support, query, source))

    def __del__(self):
        '''
            Need to terminate the processes when deleting the object
        '''
        for i in range(self.num_cores):
            self.p_list[i].terminate()

        del self.done_queue


class ParallelSampler_Test():

    def __init__(self, data, args, num_episodes=None):
        self.data = data
        self.args = args
        self.num_episodes = num_episodes

        self.all_classes = np.unique(self.data['label'])
        self.num_classes = len(self.all_classes)
        if self.num_classes < self.args.way:
            raise ValueError("Total number of classes is less than #way.")

        self.idx_list = []
        for y in self.all_classes:
            self.idx_list.append(
                    np.squeeze(np.argwhere(self.data['label'] == y)))

        self.count = 0
        self.done_queue = Queue()

        self.num_cores = cpu_count() if args.n_workers is 0 else args.n_workers

        self.p_list = []
        for i in range(self.num_cores):
            self.p_list.append(
                    Process(target=self.worker, args=(self.done_queue,)))

        for i in range(self.num_cores):
            self.p_list[i].start()

    def get_epoch(self):
        for _ in range(self.num_episodes):
            # wait until self.thread finishes
            support, query = self.done_queue.get()

            # convert to torch.tensor
            support = utils.to_tensor(support, self.args.cuda, ['raw'])
            query = utils.to_tensor(query, self.args.cuda, ['raw'])

            support['is_support'] = True
            query['is_support'] = False

            yield support, query

    def worker(self, done_queue):
        '''
            Generate one task (support and query).
            Store into self.support[self.cur] and self.query[self.cur]
        '''
        while True:
            if done_queue.qsize() > 100:
                time.sleep(1)
                continue
            # sample ways
            sampled_classes = np.random.permutation(
                    self.num_classes)[:self.args.way]

            source_classes = []
            for j in range(self.num_classes):
                if j not in sampled_classes:
                    source_classes.append(self.all_classes[j])
            source_classes = sorted(source_classes)

            # sample examples
            support_idx, query_idx = [], []
            for y in sampled_classes:
                tmp = np.random.permutation(len(self.idx_list[y]))
                support_idx.append(
                        self.idx_list[y][tmp[:self.args.shot]])
                query_idx.append(
                        self.idx_list[y][
                            tmp[self.args.shot:self.args.shot+self.args.query]])

            support_idx = np.concatenate(support_idx)
            query_idx = np.concatenate(query_idx)
            if self.args.mode == 'finetune' and len(query_idx) == 0:
                query_idx = support_idx

            # aggregate examples
            max_support_len = np.max(self.data['text_len'][support_idx])
            max_query_len = np.max(self.data['text_len'][query_idx])

            support = utils.select_subset(self.data, {}, ['text', 'text_len', 'label'],
                                     support_idx, max_support_len)
            query = utils.select_subset(self.data, {}, ['text', 'text_len', 'label'],
                                   query_idx, max_query_len)

            done_queue.put((support, query))

    def __del__(self):
        '''
            Need to terminate the processes when deleting the object
        '''
        for i in range(self.num_cores):
            self.p_list[i].terminate()

        del self.done_queue


def train_one(task, model, optG, optD, args, grad):
    '''
        Train the model on one sampled task.
    '''
    model['G'].train()
    model['D'].train()
    model['clf'].train()

    support, query, source = task
    for _ in range(args.k):
        # ***************update D**************
        optD.zero_grad()

        # Embedding the document
        XS, XS_inputD, _ = model['G'](support, flag='support')
        YS = support['label']
        # print('YS', YS)

        XQ, XQ_inputD, _ = model['G'](query, flag='query')
        YQ = query['label']
        YQ_d = torch.ones(query['label'].shape, dtype=torch.long).to(query['label'].device)
        # print('YQ', set(YQ.numpy()))

        XSource, XSource_inputD, _ = model['G'](source, flag='query')
        YSource_d = torch.zeros(source['label'].shape, dtype=torch.long).to(source['label'].device)

        XQ_logitsD = model['D'](XQ_inputD)
        XSource_logitsD = model['D'](XSource_inputD)

        d_loss = F.cross_entropy(XQ_logitsD, YQ_d) + F.cross_entropy(XSource_logitsD, YSource_d)
        d_loss.backward(retain_graph=True)
        grad['D'].append(get_norm(model['D']))
        optD.step()

        # *****************update G****************
        optG.zero_grad()
        XQ_logitsD = model['D'](XQ_inputD)
        XSource_logitsD = model['D'](XSource_inputD)
        d_loss = F.cross_entropy(XQ_logitsD, YQ_d) + F.cross_entropy(XSource_logitsD, YSource_d)

        acc, d_acc, loss, _ = model['clf'](XS, YS, XQ, YQ, XQ_logitsD, XSource_logitsD, YQ_d, YSource_d)

        g_loss = loss - d_loss
        if args.ablation == "-DAN":
            g_loss = loss
            print("%%%%%%%%%%%%%%%%%%%This is ablation mode: -DAN%%%%%%%%%%%%%%%%%%%%%%%%%%")
        g_loss.backward(retain_graph=True)
        grad['G'].append(get_norm(model['G']))
        grad['clf'].append(get_norm(model['clf']))
        optG.step()

    return d_acc


def train(train_data, val_data, model, args):
    '''
        Train the model
        Use val_data to do early stopping
    '''
    # creating a tmp directory to save the models
    out_dir = os.path.abspath(os.path.join(
                                  os.path.curdir,
                                  "tmp-runs",
                                  str(int(time.time() * 1e7))))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    best_acc = 0
    sub_cycle = 0
    best_path = None

    optG = torch.optim.Adam(grad_param(model, ['G', 'clf']), lr=args.lr_g)
    optD = torch.optim.Adam(grad_param(model, ['D']), lr=args.lr_d)

    if args.lr_scheduler == 'ReduceLROnPlateau':
        schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optG, 'max', patience=args.patience//2, factor=0.1, verbose=True)
        schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optD, 'max', patience=args.patience // 2, factor=0.1, verbose=True)

    elif args.lr_scheduler == 'ExponentialLR':
        schedulerG = torch.optim.lr_scheduler.ExponentialLR(optG, gamma=args.ExponentialLR_gamma)
        schedulerD = torch.optim.lr_scheduler.ExponentialLR(optD, gamma=args.ExponentialLR_gamma)



    print("{}, Start training".format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')), flush=True)

    # train_gen = ParallelSampler(train_data, args, args.train_episodes)
    train_gen_val = ParallelSampler_Test(train_data, args, args.val_episodes)
    val_gen = ParallelSampler_Test(val_data, args, args.val_episodes)

    # sampled_classes, source_classes = task_sampler(train_data, args)
    for ep in range(args.train_epochs):

        sampled_classes, source_classes = task_sampler(train_data, args)

        train_gen = ParallelSampler(train_data, args, sampled_classes, source_classes, args.train_episodes)

        sampled_tasks = train_gen.get_epoch()

        grad = {'clf': [], 'G': [], 'D': []}

        if not args.notqdm:
            sampled_tasks = tqdm(sampled_tasks, total=train_gen.num_episodes,
                    ncols=80, leave=False, desc=colored('Training on train',
                        'yellow'))
        d_acc = 0
        for task in sampled_tasks:
            if task is None:
                break
            d_acc += train_one(task, model, optG, optD, args, grad)

        d_acc = d_acc / args.train_episodes

        print("---------------ep:" + str(ep) + " d_acc:" + str(d_acc) + "-----------")

        if ep % 10 == 0:

            acc, std, _ = test(train_data, model, args, args.val_episodes, False,
                            train_gen_val.get_epoch())
            print("{}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f} ".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                "ep", ep,
                colored("train", "red"),
                colored("acc:", "blue"), acc, std,
                ), flush=True)

        # Evaluate validation accuracy
        cur_acc, cur_std, _ = test(val_data, model, args, args.val_episodes, False,
                                val_gen.get_epoch())
        print(("{}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f}, "
               "{:s} {:s}{:>7.4f}, {:s}{:>7.4f}").format(
               datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
               "ep", ep,
               colored("val  ", "cyan"),
               colored("acc:", "blue"), cur_acc, cur_std,
               colored("train stats", "cyan"),
               colored("G_grad:", "blue"), np.mean(np.array(grad['G'])),
               colored("clf_grad:", "blue"), np.mean(np.array(grad['clf'])),
               ), flush=True)

        # Update the current best model if val acc is better
        if cur_acc > best_acc:
            best_acc = cur_acc
            best_path = os.path.join(out_dir, str(ep))

            # save current model
            print("{}, Save cur best model to {}".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                best_path))

            torch.save(model['G'].state_dict(), best_path + '.G')
            torch.save(model['D'].state_dict(), best_path + '.D')
            torch.save(model['clf'].state_dict(), best_path + '.clf')

            sub_cycle = 0
        else:
            sub_cycle += 1

        # Break if the val acc hasn't improved in the past patience epochs
        if sub_cycle == args.patience:
            break

        if args.lr_scheduler == 'ReduceLROnPlateau':
            schedulerG.step(cur_acc)
            schedulerD.step(cur_acc)

        elif args.lr_scheduler == 'ExponentialLR':
            schedulerG.step()
            schedulerD.step()

    print("{}, End of training. Restore the best weights".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')),
            flush=True)

    # restore the best saved model
    model['G'].load_state_dict(torch.load(best_path + '.G'))
    model['D'].load_state_dict(torch.load(best_path + '.D'))
    model['clf'].load_state_dict(torch.load(best_path + '.clf'))

    if args.save:
        # save the current model
        out_dir = os.path.abspath(os.path.join(
                                      os.path.curdir,
                                      "saved-runs",
                                      str(int(time.time() * 1e7))))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        best_path = os.path.join(out_dir, 'best')

        print("{}, Save best model to {}".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
            best_path), flush=True)

        torch.save(model['G'].state_dict(), best_path + '.G')
        torch.save(model['D'].state_dict(), best_path + '.D')
        torch.save(model['clf'].state_dict(), best_path + '.clf')

        with open(best_path + '_args.txt', 'w') as f:
            for attr, value in sorted(args.__dict__.items()):
                f.write("{}={}\n".format(attr, value))

    return


def test_one(task, model, args):
    '''
        Evaluate the model on one sampled task. Return the accuracy.
    '''
    support, query = task
    # print("query_text.shape:", query['text'].shape)

    if args.embedding != 'mlad':

        # Embedding the document
        XS = model['ebd'](support)
        YS = support['label']

        XQ = model['ebd'](query)
        YQ = query['label']

        # Apply the classifier
        acc, _ = model['clf'](XS, YS, XQ, YQ)

        return acc

    else:
        # Embedding the document
        XS, XS_inputD, XS_avg = model['G'](support, flag='support')
        YS = support['label']
        # print('YS', YS)

        XQ, XQ_inputD, XQ_avg = model['G'](query, flag='query')
        YQ = query['label']
        YQ_d = torch.ones(query['label'].shape, dtype=torch.long).to(query['label'].device)
        # print('YQ', set(YQ.numpy()))

        # 这步主要是为了匹配模型输入，下面这几个参数没有什么用
        XSource, XSource_inputD, _ = model['G'](query, flag='query')
        YSource_d = torch.zeros(query['label'].shape, dtype=torch.long).to(query['label'].device)

        XQ_logitsD = model['D'](XQ_inputD)
        XSource_logitsD = model['D'](XSource_inputD)

        # 把原始数据变成长度都为50，为了可视化的，不影响最终结果
        query_data = query['text']
        if query_data.shape[1] < 50:
            zero = torch.zeros((query_data.shape[0], 50 - query_data.shape[1]))
            if args.cuda != -1:
                zero = zero.cuda(args.cuda)
            query_data = torch.cat((query_data, zero), dim=-1)
            # print('reverse_feature.shape[1]', reverse_feature.shape[1])
        else:
            query_data = query_data[:, :50]
            # print('reverse_feature.shape[1]', reverse_feature.shape[1])

        # Apply the classifier
        acc, d_acc, loss, x_hat = model['clf'](XS, YS, XQ, YQ, XQ_logitsD, XSource_logitsD, YQ_d, YSource_d, query_data)

        all_sentence_ebd = XQ
        all_avg_sentence_ebd = XQ_avg
        all_label = YQ
        # print(all_sentence_ebd.shape, all_avg_sentence_ebd.shape, all_label.shape)


        return acc, d_acc, all_sentence_ebd.cpu().detach().numpy(), all_avg_sentence_ebd.cpu().detach().numpy(), all_label.cpu().detach().numpy(), XQ_inputD.cpu().detach().numpy(), query_data.cpu().detach().numpy(), x_hat.cpu().detach().numpy()


def test(test_data, model, args, num_episodes, verbose=True, sampled_tasks=None):
    '''
        Evaluate the model on a bag of sampled tasks. Return the mean accuracy
        and its std.
    '''
    model['G'].eval()
    model['D'].eval()
    model['clf'].eval()

    if sampled_tasks is None:
        sampled_tasks = ParallelSampler_Test(test_data, args,
                                        num_episodes).get_epoch()

    acc = []
    d_acc = []
    all_sentence_ebd = None
    all_avg_sentence_ebd = None
    all_sentence_label = None
    all_word_weight = None
    all_query_data = None
    all_x_hat = None
    all_drawn_data = {}
    if not args.notqdm:
        sampled_tasks = tqdm(sampled_tasks, total=num_episodes, ncols=80,
                             leave=False,
                             desc=colored('Testing on val', 'yellow'))
    count = 0
    for task in sampled_tasks:
        if args.embedding == 'mlad':
            acc1, d_acc1, sentence_ebd, avg_sentence_ebd, sentence_label, word_weight, query_data, x_hat = test_one(task, model, args)
            if count < 20:
                if all_sentence_ebd is None:
                    all_sentence_ebd = sentence_ebd
                    all_avg_sentence_ebd = avg_sentence_ebd
                    all_sentence_label = sentence_label
                    all_word_weight = word_weight
                    all_query_data = query_data
                    all_x_hat = x_hat
                else:
                    all_sentence_ebd = np.concatenate((all_sentence_ebd, sentence_ebd), 0)
                    all_avg_sentence_ebd = np.concatenate((all_avg_sentence_ebd, avg_sentence_ebd), 0)
                    all_sentence_label = np.concatenate((all_sentence_label, sentence_label))
                    all_word_weight = np.concatenate((all_word_weight, word_weight), 0)
                    all_query_data = np.concatenate((all_query_data, query_data), 0)
                    all_x_hat = np.concatenate((all_x_hat, x_hat), 0)
            count = count + 1
            acc.append(acc1)
            d_acc.append(d_acc1)
        else:
            acc.append(test_one(task, model, args))

    acc = np.array(acc)
    d_acc = np.array(d_acc)
    # all_drawn_data["sentence_ebd"] = all_sentence_ebd.tolist()
    # all_drawn_data["avg_sentence_ebd"] = all_avg_sentence_ebd.tolist()
    # all_drawn_data["label"] = all_sentence_label.tolist()
    # all_drawn_data["word_weight"] = all_word_weight.tolist()
    # all_drawn_data["query_data"] = all_query_data.tolist()
    all_x = []
    for _x in all_x_hat.tolist():
        x_ = []
        for x_x in _x:
            x_.append(args.id2word[x_x])
        all_x.append(x_)
    all_drawn_data["x_hat"] = all_x


    if verbose:
        if args.embedding != 'mlad':
            print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                colored("test acc mean", "blue"),
                np.mean(acc),
                colored("test std", "blue"),
                np.std(acc),
                ), flush=True)
        else:
            print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                colored("test acc mean", "blue"),
                np.mean(acc),
                colored("test std", "blue"),
                np.std(acc),
                colored("test d_acc mean", "blue"),
                np.mean(d_acc),
                colored("test d_acc std", "blue"),
                np.std(d_acc),
            ), flush=True)

    return np.mean(acc), np.std(acc), all_drawn_data


# def Drawn_Query_Vector(test_data, model, args):
#     """
#     Visualization: Drawn query vector by TSNE or PCA.
#     """
#     test_data['text'] = torch.from_numpy(test_data['text']).cuda()
#     label = test_data['label']
#     sentence_ebd, _, avg_sentence_ebd = model['G'](test_data)
#     data_drawn = {}
#     data_drawn["sentence_ebd"] = sentence_ebd
#     data_drawn["avg_sentence_ebd"] = avg_sentence_ebd
#     data_drawn["label"] = label
#     path = args.path_drawn_data
#     with open(path, 'a') as f_w:
#         f_w.write(data_drawn)
#         f_w.flush()
#         f_w.close()



def to_tensor(data, cuda, exclude_keys=[]):
    '''
        Convert all values in the data into torch.tensor
    '''
    for key in data.keys():
        if key in exclude_keys:
            continue

        data[key] = torch.from_numpy(data[key]).to(torch.int64)
        if cuda != -1:
            data[key] = data[key].cuda(cuda)

    return data


def main():

    # make_print_to_file(path='/results')

    args = parse_args()

    print_args(args)

    set_seed(args.seed)

    # load data
    train_data, val_data, test_data, vocab = loader.load_dataset(args)

    args.id2word = vocab.itos

    # initialize model
    model = {}
    model["G"], model["D"] = get_embedding(vocab, args)
    model["clf"] = get_classifier(model["G"].ebd_dim, args)

    if args.mode == "train":
        # train model on train_data, early stopping based on val_data
        train(train_data, val_data, model, args)

    val_acc, val_std, _ = test(val_data, model, args,
                                            args.val_episodes)

    test_acc, test_std, drawn_data = test(test_data, model, args,
                                          args.test_episodes)

    path_drawn = args.path_drawn_data
    with open(path_drawn, 'w') as f_w:
        json.dump(drawn_data, f_w)
        print("store drawn data finished.")

    # file_path = r'../data/attention_data.json'
    # Print_Attention(file_path, vocab, model, args)

    if args.result_path:
        directory = args.result_path[:args.result_path.rfind("/")]
        if not os.path.exists(directory):
            os.mkdirs(directory)

        result = {
            "test_acc": test_acc,
            "test_std": test_std,
            "val_acc": val_acc,
            "val_std": val_std
        }

        for attr, value in sorted(args.__dict__.items()):
            result[attr] = value

        with open(args.result_path, "wb") as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


def Print_Attention(file_path, vocab, model, args):
    model['G'].eval()
    word2id = vocab.itos

    data = []
    for line in open(file_path, 'r'):
        data.append(json.loads(line))

    output = {}
    output['text'] = []
    for i, temp in enumerate(data):
        output['text'].append(temp['text'])


    for i, temp in enumerate(data):
        tem = []
        length = len(temp['text'])
        for word in temp['text']:
            if word in word2id:
                tem.append(word2id.index(word))
        data[i]['text'] = np.array(tem)
        data[i]['text_len'] = 20

    data2 = {}
    data2['text'] = []
    data2['text_len'] = []
    data2['label'] = []
    for i, temp in enumerate(data):
        # 将temp['text']统一变为[20]，长则截断，短则补0
        if temp['text'].shape[0] < 200:
            zero = torch.zeros(20 - temp['text'].shape[0])
            temp['text'] = np.concatenate((temp['text'], zero))
        else:
            temp['text'] = temp['text'][:20]
        data2['text'].append(temp['text'])
        data2['text_len'].append(temp['text_len'])
        data2['label'].append(temp['label'])

    data2['text'] = np.array(data2['text'])
    data2['text_len'] = np.array(data2['text_len'])
    data2['label'] = np.array(data2['label'])

    query = to_tensor(data2, args.cuda)
    query['is_support'] = False

    XQ, XQ_inputD, XQ_avg = model['G'](query, flag='query')
    output['attention'] = []
    for i, temp in enumerate(data):
        output['attention'].append(XQ_inputD[i].cpu().detach().numpy().tolist())
    output_file_path = 'output_attention.json'
    f_w = open(output_file_path, 'w')
    f_w.write(json.dumps(output))
    f_w.flush()
    f_w.close()


def load_model_state_dict(model, model_path):
    # 初始化模型参数
    model_dict = model.state_dict()                                    # 取出自己网络的参数字典
    pretrained_dict = torch.load(model_path)# 加载预训练网络的参数字典
    # 取出预训练网络的参数字典
    keys = []
    for k, v in pretrained_dict.items():
           keys.append(k)

    i = 0

    # 自己网络和预训练网络结构一致的层，使用预训练网络对应层的参数初始化
    print("_____________pretrain_parameters______________________________")
    for k, v in model_dict.items():
        if v.size() == pretrained_dict[keys[i]].size():
            model_dict[k] = pretrained_dict[keys[i]]
            print(model_dict[k])
            i = i + 1
        # print(model_dict[k])
    print("___________________________________________________________")
    model.load_state_dict(model_dict)
    return model


def main_attention():

    args = parse_args()

    print_args(args)

    set_seed(args.seed)

    # load data
    train_data, val_data, test_data, vocab = loader.load_dataset(args)

    # initialize model
    model = {}
    model["G"], model["D"] = get_embedding(vocab, args)
    model["clf"] = get_classifier(model["G"].ebd_dim, args)

    best_path = '../bin/tmp-runs/16116280768954578/18'
    model['G'].load_state_dict(torch.load(best_path + '.G'))
    # model['D'].load_state_dict(torch.load(best_path + '.D'))
    # model['clf'].load_state_dict(torch.load(best_path + '.clf'))

    # if args.pretrain is not None:
    #     model["ebd"] = load_model_state_dict(model["G"], args.pretrain)

    file_path = r'../data/attention_data.json'
    Print_Attention(file_path, vocab, model, args)



if __name__ == '__main__':
    main()