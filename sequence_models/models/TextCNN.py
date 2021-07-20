"""
Author:
    Xiaoqiang Zhang, 1365677361@qq.com

Reference:
    [1] Convolutional Neural Networks for Sentence Classification[J]. arXiv:1408.5882, 2014.
       (https://arxiv.org/abs/1408.5882)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel


class TextCNN(BaseModel):
    def __init__(self, embed_num, embed_dim=128, class_num=2, kernel_num=100, kernel_sizes=(2, 3, 4),
                 dropout=0.5):
        super(TextCNN, self).__init__()
        V = embed_num
        D = embed_dim
        C = class_num
        Ci = 1
        Co = kernel_num
        Ks = kernel_sizes

        self.embedding = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (2, D))
        self.conv14 = nn.Conv2d(Ci, Co, (3, D))
        self.conv15 = nn.Conv2d(Ci, Co, (4, D))
        '''
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, 1)
        self.act = nn.Sigmoid()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, seqs, seqs_lengths):
        embedded = self.embedding(seqs)   # (N, W, D)
        x = embedded.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        outputs = self.act(logit)
        return outputs, embedded
