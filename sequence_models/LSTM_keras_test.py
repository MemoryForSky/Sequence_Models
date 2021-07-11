import warnings
import numpy as np
import pandas as pd
from models.Seq2Vec import Seq2Vec
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import *
from models.LSTM_keras import SeqLSTM

warnings.filterwarnings('ignore')
seed = 2021

EMBED_SIZE = 64
MAX_SEQUENCE_LENGTH = 128

# %%
train = pd.read_csv('./data/train_seqs.csv')
test = pd.read_csv('./data/test_seqs.csv')

data = pd.concat([train, test])
data['tagid'] = data.tagid.apply(lambda x: x.split(' '))
seqs = list(data.tagid.values)
MAX_NB_WORDS = len(set([i for seq in seqs for i in seq]))
print(f"数据中共有标签{MAX_NB_WORDS}个")

# %%
seqlstm = SeqLSTM(seqs, embed_size=EMBED_SIZE, max_sequence_length=MAX_SEQUENCE_LENGTH)
seqs_token = seqlstm.tokenizer(MAX_NB_WORDS)
embedding_matrix = seqlstm.build_pre_emb()

# %%
# 划分训练集和测试集
X_train = seqs_token[:train.shape[0]]
X_test = seqs_token[train.shape[0]:]
y = train['label'].values

# %%
# 五折交叉验证
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
oof = np.zeros([len(train), 1])
predictions = np.zeros([len(test), 1])

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
    print("fold n{}".format(fold_ + 1))
    model = seqlstm.LSTM(embedding_matrix)
    if fold_ == 0:
        model.summary()

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)
    bst_model_path = "./outputs/{}.h5".format(fold_)
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    X_tra, X_val = X_train[trn_idx], X_train[val_idx]
    y_tra, y_val = y[trn_idx], y[val_idx]

    model.fit(X_tra, y_tra,
              validation_data=(X_val, y_val),
              epochs=1, batch_size=256, shuffle=True,
              callbacks=[early_stopping, model_checkpoint])

    model.load_weights(bst_model_path)

    oof[val_idx] = model.predict(X_val)

    predictions += model.predict(X_test) / folds.n_splits
    print(predictions)
    del model

print("AUC score: {}".format(roc_auc_score(y, oof)))
print("F1 score: {}".format(f1_score(y, [1 if i >= 0.5 else 0 for i in oof])))
print("Precision score: {}".format(precision_score(y, [1 if i >= 0.5 else 0 for i in oof])))
print("Recall score: {}".format(recall_score(y, [1 if i >= 0.5 else 0 for i in oof])))

"""
AUC score: 0.7587734087111111
F1 score: 0.6885398984155358
Precision score: 0.6888868127248097
Recall score: 0.6881933333333333
"""
