# 在正式训练之前，我们先用训练集和验证集对模型的G生成器和D判别器进行初始化
# 初始化方法就是直接把他们当成一个端到端的网络进行文本分类任务

# 这个文件的预训练有问题，就是在sample阶段，每次sample不一样的类，但是到了预测时都会把标签映射成0-x，这样会搞蒙模型。

import os
import sys
import time
import datetime
import pickle
import signal
import argparse
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def reidx_y(YS, YQ, args):
        '''
            Map the labels into 0,..., way
            @param YS: batch_size
            @param YQ: batch_size

            @return YS_new: batch_size
            @return YQ_new: batch_size
        '''
        unique1, inv_S = torch.unique(YS, sorted=True, return_inverse=True)
        unique2, inv_Q = torch.unique(YQ, sorted=True, return_inverse=True)

        Y_new = torch.arange(start=0, end=args.way, dtype=unique1.dtype,
                device=unique1.device)

        return Y_new[inv_S], Y_new[inv_Q]


def train(train_data, model, args):
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

    opt = torch.optim.Adam(grad_param(model, ['ebd']), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, 'max', patience=args.patience // 2, factor=0.1, verbose=True)

    print("{}, Start training".format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')), flush=True)

    train_gen = ParallelSampler(train_data, args, args.train_episodes)  # 这里每次选择5个类，每个类中抽取一些样本进行训练，或许可以改为一个完全端到端的预测，就是只选择5个类，用这5个类的数据去预训练模型。
    train_gen_val = ParallelSampler(train_data, args, args.val_episodes)
    # val_gen = ParallelSampler(val_data, args, args.val_episodes)

    for ep in range(args.train_epochs):
        sampled_tasks = train_gen.get_epoch()

        grad = {'ebd': []}

        if not args.notqdm:
            sampled_tasks = tqdm(sampled_tasks, total=train_gen.num_episodes,
                                 ncols=80, leave=False, desc=colored('Training on train',
                                                                     'yellow'))

        for task in sampled_tasks:
            if task is None:
                break
            train_one(task, model, opt, args, grad)

        acc, std = test(train_data, model, args, args.val_episodes, False,
                        train_gen_val.get_epoch())
        print("{}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f} ".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
            "ep", ep,
            colored("train", "red"),
            colored("acc:", "blue"), acc, std,
        ), flush=True)

        if acc >= best_acc:
            best_acc = acc
            best_path = os.path.join(out_dir, str(ep))

            print("{}, Save cur best model to {}".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                best_path))

            torch.save(model['ebd'].state_dict(), best_path + '.pretrain_ebd')

            sub_cycle = 0

        else:
            sub_cycle += 1

        # Break if the val acc hasn't improved in the past patience epochs
        if sub_cycle == args.patience:
            break

    print("{}, End of training. Restore the best weights".format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')),
        flush=True)

    # restore the best saved model
    model['ebd'].load_state_dict(torch.load(best_path + '.pretrain_ebd'))
    # model['clf'].load_state_dict(torch.load(best_path + '.clf'))

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

        torch.save(model['ebd'].state_dict(), best_path + '.pretrain_ebd')
        # torch.save(model['clf'].state_dict(), best_path + '.clf')

        with open(best_path + '_args.txt', 'w') as f:
            for attr, value in sorted(args.__dict__.items()):
                f.write("{}={}\n".format(attr, value))

    return


def train_one(task, model, opt, args, grad):
    '''
        Train the model on one sampled task.
    '''
    model['ebd'].train()
    opt.zero_grad()

    support, query = task

    if args.embedding != 'mlad':
        # Embedding the document
        XS = model['ebd'](support)
        YS = support['label']

        XQ = model['ebd'](query)
        YQ = query['label']

        # Apply the classifier
        _, loss = model['clf'](XS, YS, XQ, YQ)

    else:
        # Embedding the document
        XS, d_logits_S = model['ebd'](support, flag='support')
        YS = support['label']
        # print('\nYS', YS)

        XQ, d_logits_Q = model['ebd'](query, flag='query')
        YQ = query['label']
        # print('\nYQ', YQ)
        #
        # print('\nd_s', d_logits_S.shape)
        # print('\nd_q', d_logits_Q.shape)

        YS, YQ = reidx_y(YS, YQ, args)

        loss = F.cross_entropy(d_logits_S, YS) + F.cross_entropy(d_logits_Q, YQ)
        # print("______________________________", loss)


    if loss is not None:

        loss.backward()

    if torch.isnan(loss):
        # do not update the parameters if the gradient is nan
        print("NAN detected")
        # print(model['clf'].lam, model['clf'].alpha, model['clf'].beta)
        return

    if args.clip_grad is not None:
        nn.utils.clip_grad_value_(grad_param(model, ['ebd']),
                                  args.clip_grad)

    grad['ebd'].append(get_norm(model['ebd']))

    opt.step()


def test(test_data, model, args, num_episodes, verbose=True, sampled_tasks=None):
    '''
        Evaluate the model on a bag of sampled tasks. Return the mean accuracy
        and its std.
    '''
    model['ebd'].eval()

    if sampled_tasks is None:
        sampled_tasks = ParallelSampler(test_data, args,
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
        XS, d_logits_S = model['ebd'](support, flag='support')
        YS = support['label']

        XQ, d_logits_Q = model['ebd'](query, flag='query')
        YQ = query['label']

        # Apply the classifier
        d_acc = (base.compute_acc(d_logits_Q, YQ) + base.compute_acc(d_logits_S, YS)) / 2

        acc = d_acc

        return acc, d_acc



def main():

    args = parse_args()

    print_args(args)

    set_seed(args.seed)

    # load data
    train_data, val_data, test_data, vocab = loader.load_dataset(args)

    print(train_data['avg_ebd'], val_data['avg_ebd'])

    # initialize model
    model = {}
    model["ebd"] = ebd.get_embedding(vocab, args)

    train(train_data, model, args)


if __name__ == '__main__':
    main()

