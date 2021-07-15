import numpy as np
import torch
import torch.nn.functional as F
from torchtext.legacy import data
import math
import time

SEED = 123
BATCH_SIZE = 128
LEARNING_RATE = 1e-3  # 学习率
EMBEDDING_DIM = 100  # 词向量维度

# 设置device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 为CPU设置随机种子
torch.manual_seed(123)

# 两个Field对象定义字段的处理方法（文本字段、标签字段）
TEXT = data.Field(tokenize=lambda x: x.split(), lower=True)
LABEL = data.LabelField(dtype=torch.float)


# get_dataset: 返回Dataset所需的 text 和 label
def get_dataset(corpus_path, text_field, label_field):
    fields = [('text', text_field), ('label', label_field)]  # torchtext文本配对关系
    examples = []

    with open(corpus_path) as f:
        li = []
        while True:
            content = f.readline().replace('\n', '')
            if not content:  # 为空行，表示取完一次数据(一次的数据保存在li中)
                if not li:
                    break
                label = li[0][10]
                text = li[1][6:-7]
                examples.append(data.Example.fromlist([text, label], fields=fields))
                li = []
            else:
                li.append(content)
    return examples, fields


# 得到构建Dataset所需的examples 和 fields
train_examples, train_fileds = get_dataset('./data/trains.txt', TEXT, LABEL)
dev_examples, dev_fields = get_dataset('./data/dev.txt', TEXT, LABEL)
test_examples, test_fields = get_dataset('./data/tests.txt', TEXT, LABEL)

# 构建Dataset数据集
train_data = data.Dataset(train_examples, train_fileds)
dev_data = data.Dataset(dev_examples, dev_fields)
test_data = data.Dataset(test_examples, test_fields)
# for t in test_data:
#     print(t.text, t.label)

print('len of train data:', len(train_data))  # 1000
print('len of dev data:', len(dev_data))  # 200
print('len of test data:', len(test_data))  # 300

# 创建vocabulary
TEXT.build_vocab(train_data, max_size=5000, vectors='glove.6B.100d')
LABEL.build_vocab(train_data)
print(len(TEXT.vocab))  # 3287
print(TEXT.vocab.itos[:12])  # ['<unk>', '<pad>', 'the', 'and', 'a', 'to', 'is', 'was', 'i', 'of', 'for', 'in']
print(TEXT.vocab.stoi['love'])  # 129
# print(TEXT.vocab.stoi)         # defaultdict {'<unk>': 0, '<pad>': 1, ....}

# 创建iterators, 每个iteration都会返回一个batch的example
train_iterator, dev_iterator, test_iterator = data.BucketIterator.splits(
                                                        (train_data, dev_data, test_data),
                                                        batch_size=BATCH_SIZE,
                                                        device=device,
                                                        sort=False)
