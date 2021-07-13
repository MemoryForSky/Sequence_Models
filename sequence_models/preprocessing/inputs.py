import torch
from torchtext.legacy import data


def data_preprocessing(train_path, test_path, tokenize='spacy'):
    TEXT = data.Field(tokenize=tokenize, batch_first=True, include_lengths=True)
    LABEL = data.LabelField(dtype=torch.float, batch_first=True)
    fields = [('id', None), ('text', TEXT), ('label', LABEL)]
    training_data = data.TabularDataset(path=train_path,
                                        format='csv',
                                        fields=fields,
                                        skip_header=True)
    test_datafields = [('id', None), ('text', TEXT)]
    test_data = data.TabularDataset(path=test_path,
                                    format='csv',
                                    skip_header=True,
                                    fields=test_datafields)
    TEXT.build_vocab(training_data, min_freq=3, vectors="glove.6B.100d")
    LABEL.build_vocab(training_data)
    size_of_vocab = len(TEXT.vocab)
    return training_data, test_data, TEXT, size_of_vocab
