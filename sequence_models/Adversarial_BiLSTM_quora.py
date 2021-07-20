import warnings
from torchtext.legacy import data
from preprocessing.inputs import data_preprocessing
from models.Adversarial_BiLSTM import AdvBiLSTM
warnings.filterwarnings('ignore')

# %%
# 1.数据预处理
train_dir = './data/quora_train.csv'
test_dir = './data/quora_test.csv'
training_data, test_data, TEXT, vocab_size = data_preprocessing(train_dir, test_dir, tokenize='spacy')
print('data examples: ', vars(training_data.examples[0]))

# %%
# 2.模型训练
model = AdvBiLSTM(vocab_size, embedding_dim=100, batch_size=64)
# 初始化预训练embedding
# pretrained_embeddings = TEXT.vocab.vectors
# model.embedding.weight.data.copy_(pretrained_embeddings)
# model.embedding.weight.requires_grad = False
model.compile("adam", "binary_cross_entropy", metrics=['auc', 'accuracy'],)
model.fit(training_data, split_ratio=0.1, epochs=10, do_validation=True)

# %%
# 3.预测
test_iter = data.Iterator(dataset=test_data,
                          batch_size=64,
                          sort_key=lambda x: len(x.text),
                          sort_within_batch=True)
predictions = model.predict(test_iter)
print('data predictions: ', predictions[:10])
