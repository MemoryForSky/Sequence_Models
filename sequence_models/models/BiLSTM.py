import torch
from torch import nn
from models.base_model import BaseModel


class BiLSTM(BaseModel):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=32, output_dim=1, n_layers=2,
                 bidirectional=True, dropout=0.2, batch_size=64):
        super(BiLSTM, self).__init__(batch_size=batch_size)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Sigmoid()

    def forward(self, text, text_lengths):
        """text = [batch size, sent_length]"""
        embedded = self.embedding(text)    # embedded = [batch size, sent_len, emb dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [batch size, num layers * num directions, hid dim]
        # cell = [batch size, num layers * num directions, hid dim]

        # 连接最后的正向和反向隐状态
        # hidden = [batch_size, hid_dim * num_directions]
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        output = self.dropout(hidden)
        dense_outputs = self.fc(output)
        outputs = self.act(dense_outputs)

        return outputs
