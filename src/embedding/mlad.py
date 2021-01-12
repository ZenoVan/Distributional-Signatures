import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding.meta import RNN
from embedding.GRL import ReverseLayerF


class MLAD(nn.Module):

    def __init__(self, ebd, args):
        super(MLAD, self).__init__()

        self.args = args

        self.ebd = ebd
        # self.aux = get_embedding(args)

        self.ebd_dim = self.ebd.embedding_dim

        self.rnn = RNN(300, 128, 1, True, 0)
        self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=1, batch_first=True, dropout=0)

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
                nn.Linear(64, 2),
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

        if self.args.mode != 'finetune':
            # Decriminator
            if flag == 'support':
                return sentence_ebd
            elif flag == 'query':
                # 梯度反转层
                reverse_feature = ReverseLayerF.apply(word_weight, 0.5)
                # reverse_feature = word_weight

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

                # 通过判别器
                logits = self.d(reverse_feature)  # [b, 500] -> [b, 2]

                return sentence_ebd, logits
        else:

            reverse_feature = ReverseLayerF.apply(word_weight, 0.5)

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
            logits = self.d(reverse_feature)  # [b, 500] -> [b, 5]

            return sentence_ebd, logits


        # ebd = torch.sum(ebd * scale, dim=1)

        # if return_score:
        #    return ebd, scale

        # return ebd