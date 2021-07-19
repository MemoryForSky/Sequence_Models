"""
Author:
    Xiaoqiang Zhang, 1365677361@qq.com

Reference:
    [1] Adversarial Training Methods For Semi-Supervised Text Classification[J]. arXiv:1605.07725, 2016.
       (https://arxiv.org/abs/1605.07725)
"""
import torch
from torch import nn
from models.base_model import BaseModel


class AdvBiLSTM(BaseModel):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=32, output_dim=1, n_layers=2,
                 bidirectional=True, dropout=0.2, batch_size=64):
        super(AdvBiLSTM, self).__init__(batch_size=batch_size)

        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # self.word_embeddings = self.word_embeddings.from_pretrained(vectors, freeze=True)

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=self.bidirectional,
                            dropout=dropout,
                            batch_first=True)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Sigmoid()

    @staticmethod
    def attention_net(self, x, query):
        d_k = query.size(-1)  # d_k为query的维度

        # scores: [batch, seq_len, seq_len] = query:[batch, seq_len, hidden_dim*2], x.t:[batch, hidden_dim*2, seq_len]
        # scores: [batch, seq_len, seq_len]
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)

        # 对最后一个维度 归一化得分
        alpha_n = F.softmax(scores, dim=-1)  # [batch, seq_len, seq_len]
        # 对权重化的x求和
        # [batch, seq_len, seq_len]·[batch,seq_len, hidden_dim*2]=[batch,seq_len,hidden_dim*2] -> [batch, hidden_dim*2]
        context = torch.matmul(alpha_n, x).sum(1)

        return context, alpha_n

    def forward(self, text, text_lengths, perturbation=None):
        """text = [batch size, sent_length]"""
        embedded = self.embedding(text)    # embedded = [batch size, sent_len, emb dim]
        if perturbation is not None:    # TODO test
            embedded += perturbation
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [batch size, num layers * num directions, hid dim]
        # cell = [batch size, num layers * num directions, hid dim]

        # 连接最后的正向和反向隐状态
        # [batch_size, seq_len, hidden_dim * 2]
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        query = self.dropout(output)
        attn_output, alpha_n = self.attention_net(output, query)
        dense_outputs = self.fc(attn_output)
        outputs = self.act(dense_outputs)

        return outputs

