from torch import nn



class SeqsLSTM(nn.Module):
    def __init__(self, seqs, nb_words=10000, embed_size=64, embedding_matrix=None, max_sequence_length=128,
                 hidden_size=128, bidirectional=False, dropout=0.2):
        super(SeqsLSTM, self).__init__()
        self.seqs = seqs
        self.nb_words = nb_words
        self.embed_size = embed_size/
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
