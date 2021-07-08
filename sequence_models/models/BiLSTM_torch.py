from torch import nn


class SeqsLSTM(nn.Module):
    def __init__(self, seqs, nb_words=10000, embed_size=64, max_sequence_length=128, embedding_matrix=None):
        super(SeqsLSTM, self).__init__()
        self.seqs = seqs
        self.nb_words = nb_words
        self.embed_size = embed_size
        self.max_sequence_length = max_sequence_length
        if embedding_matrix:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        else:
            self.embedding = nn.Embedding(self.nb_words, self.embed_size)
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=128, num_layers=1,
                            bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(128 * 2, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        embedding = self.dropout(self.embedding(input))
        output = self.lstm(embedding)
        output = self.dropout(output)
        logit = self.fc(output)
        return logit
