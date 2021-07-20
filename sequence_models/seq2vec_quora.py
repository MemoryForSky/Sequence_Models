import warnings
import numpy as np
import pandas as pd
from models.Seq2Vec import Seq2Vec
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

warnings.filterwarnings('ignore')
seed = 2021

# %%
train = pd.read_csv('./data/train_seqs.csv')
test = pd.read_csv('./data/test_seqs.csv')

data = pd.concat([train, test])
data['tagid'] = data.tagid.apply(lambda x: x.split(' '))
seqs = list(data.tagid.values)

# %%
seq2vec = Seq2Vec(emb_size=32, window=6, min_count=5, epochs=5, seed=seed)
s2v_model = seq2vec.fit(seqs)
seqs_emb = seq2vec.get_matrix(s2v_model)

# %%
features = list(seqs_emb.columns)
seqs_emb['label'] = data['label'].values
data = seqs_emb
X_train = data[~data['label'].isna()]
X_test = data[data['label'].isna()]

# %%
y = X_train['label']
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
AUC score: 0.742362847111111
F1 score: 0.6750034089737492
Precision score: 0.6734869953609376
Recall score: 0.6765266666666667
"""
