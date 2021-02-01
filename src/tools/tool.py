import argparse
import json

import torch
import numpy as np

from classifier.classifier_getter import get_classifier
from dataset import loader
from embedding.embedding import get_embedding


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
    # model options
    parser.add_argument("--embedding", type=str, default="mlad",
                        help=("document embedding method."))
    parser.add_argument("--classifier", type=str, default="r2d2",
                        help=("classifier."))
    parser.add_argument("--auxiliary", type=str, nargs="*", default=[],
                        help=("auxiliary embeddings (used for fewrel)."))
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