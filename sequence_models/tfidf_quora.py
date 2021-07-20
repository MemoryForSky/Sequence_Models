import warnings
import numpy as np
import pandas as pd
from models.TFIDF import TfIdf
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from preprocessing.inputs import data_preprocessing

warnings.filterwarnings('ignore')
seed = 2021

# %%
# 1.preprocessing
train_dir = './data/quora_train.csv'
test_dir = './data/quora_test.csv'
training_data, test_data, TEXT, vocab_size = data_preprocessing(train_dir, test_dir, tokenize='spacy')
print('data examples: ', vars(training_data.examples[0]))

# %%
train = [vars(ex)['text'] for ex in training_data.examples]
label = [vars(ex)['label'] for ex in training_data.examples]
test = [vars(ex)['text'] for ex in test_data.examples]
train = pd.DataFrame({'text': [' '.join(seq) for seq in train],
                      'label': label})
test = pd.DataFrame({'text': [' '.join(seq) for seq in test]})
data = pd.concat([train, test])
seqs = list(data.text.values)

# %%
# 2.TFIDF
tfidf = TfIdf(ngram_range=(1, 1), max_features=10000)
seqs_tfidf = tfidf.fit_transform(seqs)
seqs_dec = tfidf.decomposition(seqs_tfidf, n_components=16)

features = list(seqs_dec.columns)
seqs_dec['label'] = data['label'].values
data = seqs_dec
X_train = data[~data['label'].isna()]
X_test = data[data['label'].isna()]

# %%
# 3.classification
y = X_train['label'].astype(int)
KF = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
params = {
          'objective': 'binary',
          'metric': 'auc',
          'learning_rate': 0.05,
          'subsample': 0.8,
          'subsample_freq': 3,
          'colsample_btree': 0.8,
          'num_iterations': 10000,
          'verbose': -1
}
oof_lgb = np.zeros(len(X_train))
predictions_lgb = np.zeros((len(X_test)))
# 特征重要性
feat_imp_df = pd.DataFrame({'feat': features, 'imp': 0})

# 五折交叉验证
for fold_i, (trn_idx, val_idx) in enumerate(KF.split(X_train.values, y.values)):
    print("fold n°{}".format(fold_i))
    trn_data = lgb.Dataset(X_train.iloc[trn_idx][features], label=y.iloc[trn_idx])
    val_data = lgb.Dataset(X_train.iloc[val_idx][features], label=y.iloc[val_idx])
    num_round = 10000
    clf = lgb.train(
        params,
        trn_data,
        num_round,
        valid_sets=[trn_data, val_data],
        verbose_eval=100,
        early_stopping_rounds=50
    )
    feat_imp_df['imp'] += clf.feature_importance() / 5
    oof_lgb[val_idx] = clf.predict(X_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    predictions_lgb[:] += clf.predict(X_test[features], num_iteration=clf.best_iteration)

print("AUC score: {}".format(roc_auc_score(y, oof_lgb)))
print("F1 score: {}".format(f1_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
print("Precision score: {}".format(precision_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
print("Recall score: {}".format(recall_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))

"""
AUC score: 0.8693133907874135
F1 score: 0.789682980472308
Precision score: 0.7881246951354707
Recall score: 0.7912474401210934
"""
