import torch
from torch import nn


class SeqsLSTM(nn.Module):
    def __init__(self, seqs, nb_words=10000, embed_size=64, embedding_matrix=None, max_sequence_length=128,
                 hidden_size=128, bidirectional=False, dropout=0.2):
        super(SeqsLSTM, self).__init__()
        self.seqs = seqs
        self.nb_words = nb_words
        self.embed_size = embed_size
        self.max_sequence_length = max_sequence_length
        if embedding_matrix is None:
            self.embedding = nn.Embedding(self.nb_words, self.embed_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=hidden_size, num_layers=1,
                            bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(128 * 2, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        embedding = self.dropout(self.embedding(input))
        output = self.lstm(embedding)
        output = self.dropout(output)
        logit = self.fc(output)
        return logit


class BiLSTM(nn.Module):
    #定义所有层
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout):
        super().__init__()

        # embedding 层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # lstm 层
        self.lstm = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)

        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        # 激活函数
        self.act = nn.Sigmoid()

    def forward(self, text, text_lengths):
        # text = [batch size,sent_length]
        embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb dim]

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [batch size, num layers * num directions, hid dim]
        # cell = [batch size, num layers * num directions, hid dim]

        # 连接最后的正向和反向隐状态
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)

        # hidden = [batch size, hid dim * num directions]
        dense_outputs=self.fc(hidden)

        # 激活
        outputs=self.act(dense_outputs)

        return outputs
