"""
Author:
    Xiaoqiang Zhang, 1365677361@qq.com

Reference:
    [1] Recurrent Convolutional Neural Networks for Text Classification[J]. arXiv:1609.04243, 2016.
       (https://arxiv.org/abs/1609.04243)
"""
import torch
from torch import nn
from models.base_model import BaseModel

Config = {"hidden_size": 100,         # 字典尺寸
          "num_layer": 2,
          "bidirectiion": True,   # 双向
          "drop": 0.3,      # dropout比例
          "cnn_channel": 100,   # 1D-CNN的output_channel
          "cnn_kernel": 2,    # 1D-CNN的卷积核
          "topk": 1,  # cnn的output结果取top-k
          "fc_hidden": 10,  # 全连接层的隐藏层
          "fc_cla": 1,  # 全连接层的输出类别
          }


class RCNN(BaseModel):
    def __init__(self, vocab_size, embedding_dim=100):
        super(RCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=Config['hidden_size'],
            num_layers=Config['num_layer'],
            bidirectional=True,
            batch_first=True,
            dropout=Config['drop']
        )

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=Config['hidden_size'] * 2 + embedding_dim,  # 词向量和output维度做concat
                out_channels=Config['cnn_channel'],
                kernel_size=Config['cnn_kernel']),
            nn.BatchNorm1d(Config['cnn_channel']),
            nn.ReLU(inplace=True),

            nn.Conv1d(
                in_channels=Config['cnn_channel'],
                out_channels=Config['cnn_channel'],
                kernel_size=Config['cnn_kernel']),
            nn.BatchNorm1d(Config['cnn_channel']),
            nn.ReLU(inplace=True)

        )

        self.fc = nn.Sequential(
            nn.Linear(Config['topk'] * Config['cnn_channel'], Config['fc_hidden']),   # 2为bidirectional的拼接结果
            nn.BatchNorm1d(Config['fc_hidden']),
            nn.ReLU(inplace=True),
            nn.Linear(Config['fc_hidden'], Config['fc_cla'])

        )
        self.act = nn.Sigmoid()

    @staticmethod
    def topk_pooling(x, k, dim):
        index = torch.topk(x, k, dim=dim)[1]
        return torch.gather(x, dim=dim, index=index)

    def forward(self, seqs, seqs_lengths):
        embedded = self.embedding(seqs)
        out, _ = self.lstm(embedded)    # (B, S, 2H)
        out = torch.cat([embedded, out], dim=-1)   # (B, S, E) + (B, S, 2H) = (B, S, 2H+E)
        out = out.permute((0, 2, 1))    # (B, 2H+E, S)
        out = self.cnn(out)    # (B, C, S-m)
        x = self.topk_pooling(out, k=Config['topk'], dim=-1)   # sequence_len方向取top2，  (B, C, k)
        x = x.view((x.size(0), -1))    # (B, C*k)
        logits = self.fc(x)
        outputs = self.act(logits)
        return outputs, embedded
