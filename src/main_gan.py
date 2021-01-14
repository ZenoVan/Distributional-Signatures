import os
import sys
import time
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
from train.utils import load_model_state_dict
from train.utils import named_grad_param, grad_param, get_norm
from embedding.wordebd import WORDEBD

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# def make_print_to_file(path='./'):
#     '''
#     path， it is a path for save your log about fuction print
#     example:
#     use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
#     :return:
#     '''
#     import os
#     import sys
#     import datetime
#
#     class Logger(object):
#         def __init__(self, filename="Default.log", path="./"):
#             self.terminal = sys.stdout
#             self.log = open(os.path.join(path, filename), "a", encoding='utf8', )
#
#         def write(self, message):
#             self.terminal.write(message)
#             self.log.write(message)
#
#         def flush(self):
#             pass
#
#     fileName = datetime.datetime.now().strftime('day' + '%Y_%m_%d')
#     sys.stdout = Logger(fileName + '.log', path=path)
#
#     #############################################################
#     # 这里输出之后的所有的输出的print 内容即将写入日志
#     #############################################################
#     print(fileName.center(60, '*'))


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
    parser.add_argument("--induct_att_dim", type=int, default=64,
                        help=("attention projection dim of induction network"))

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

    parser.add_argument("--pretrain", type=str, default=None, help="path to the pretraiend weights for MLAD")
    parser.add_argument("--k", type=int, default=None, help="Number of iterations of the adversarial network")
    parser.add_argument("--lr_g", type=float, default=1e-3, help="learning rate of G")
    parser.add_argument("--lr_d", type=float, default=1e-3, help="learning rate of D")
    parser.add_argument("--lr_scheduler", type=str, default=None, help="lr_scheduler")
    parser.add_argument("--ExponentialLR_gamma", type=float, default=0.98, help="ExponentialLR_gamma")
    parser.add_argument("--train_mode", type=str, default=None, help="you can choose t_add_v or None")
    parser.add_argument("--Comments", type=str, default="random split and test_val_train", help="Comments")

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

    # source_classes = []
    # for j in range(num_classes):
    #     if j not in sampled_classes:
    #         source_classes.append(all_classes[j])
    # source_classes = sorted(source_classes)

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

    def forward(self, data, flag, return_score=False):

        # 将单词转为词向量
        ebd = self.ebd(data)
        w2v = ebd

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

        return sentence_ebd, reverse_feature


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

    def forward(self, XS, YS, XQ, YQ, XQ_logitsD, XSource_logitsD, YQ_d, YSource_d):
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

        return acc, d_acc, loss


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
        XS, XS_inputD = model['G'](support, flag='support')
        YS = support['label']
        # print('YS', YS)

        XQ, XQ_inputD = model['G'](query, flag='query')
        YQ = query['label']
        YQ_d = torch.ones(query['label'].shape, dtype=torch.long).to(query['label'].device)
        # print('YQ', set(YQ.numpy()))

        XSource, XSource_inputD = model['G'](source, flag='query')
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

        acc, d_acc, loss = model['clf'](XS, YS, XQ, YQ, XQ_logitsD, XSource_logitsD, YQ_d, YSource_d)

        g_loss = loss - d_loss
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

            acc, std = test(train_data, model, args, args.val_episodes, False,
                            train_gen_val.get_epoch())
            print("{}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f} ".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                "ep", ep,
                colored("train", "red"),
                colored("acc:", "blue"), acc, std,
                ), flush=True)

        # Evaluate validation accuracy
        cur_acc, cur_std = test(val_data, model, args, args.val_episodes, False,
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
        XS, XS_inputD = model['G'](support, flag='support')
        YS = support['label']
        # print('YS', YS)

        XQ, XQ_inputD = model['G'](query, flag='query')
        YQ = query['label']
        YQ_d = torch.ones(query['label'].shape, dtype=torch.long).to(query['label'].device)
        # print('YQ', set(YQ.numpy()))

        # 这步主要是为了匹配模型输入，下面这几个参数没有什么用
        XSource, XSource_inputD = model['G'](query, flag='query')
        YSource_d = torch.zeros(query['label'].shape, dtype=torch.long).to(query['label'].device)

        XQ_logitsD = model['D'](XQ_inputD)
        XSource_logitsD = model['D'](XSource_inputD)

        # Apply the classifier
        acc, d_acc, loss = model['clf'](XS, YS, XQ, YQ, XQ_logitsD, XSource_logitsD, YQ_d, YSource_d)

        return acc, d_acc


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
    if not args.notqdm:
        sampled_tasks = tqdm(sampled_tasks, total=num_episodes, ncols=80,
                             leave=False,
                             desc=colored('Testing on val', 'yellow'))

    for task in sampled_tasks:
        if args.embedding == 'mlad':
            acc1, d_acc1 = test_one(task, model, args)
            acc.append(acc1)
            d_acc.append(d_acc1)
        else:
            acc.append(test_one(task, model, args))

    acc = np.array(acc)
    d_acc = np.array(d_acc)

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

    return np.mean(acc), np.std(acc)



def main():

    # make_print_to_file(path='/results')

    args = parse_args()

    print_args(args)

    set_seed(args.seed)

    # load data
    train_data, val_data, test_data, vocab = loader.load_dataset(args)

    # initialize model
    model = {}
    model["G"], model["D"] = get_embedding(vocab, args)
    model["clf"] = get_classifier(model["G"].ebd_dim, args)

    if args.mode == "train":
        # train model on train_data, early stopping based on val_data
        train(train_data, val_data, model, args)

    val_acc, val_std = test(val_data, model, args,
                                            args.val_episodes)

    test_acc, test_std = test(test_data, model, args,
                                          args.test_episodes)

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


if __name__ == '__main__':
    main()