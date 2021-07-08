import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class TfIdf(object):
    """
    Tf-Idf encoder for the sequence data.

    Parameters:
    -----------
    seqs: 2d-list
        the sequence data.
    ngram_range: tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different n-grams to be extracted.
        All values of n such that min_n <= n <= max_n will be used. For example an ngram_range of (1, 1)
        means only unigrams, (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams.
        Only applies if analyzer is not callable.
    max_features: int, default=None
        If not None, build a vocabulary that only consider the top max_features ordered by term frequency
        across the corpus. This parameter is ignored if vocabulary is not None.
    """
    def __init__(self, seqs, ngram_range=(1, 1), max_features=10000):
        self.seqs = [' '.join(seq) for seq in seqs]
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.sts_tfidf = None

    def fit_transform(self):
        tfv = TfidfVectorizer(ngram_range=self.ngram_range,
                              max_features=self.max_features,
                              token_pattern=r"(?u)\b[^ ]+\b")
        self.sts_tfidf = tfv.fit_transform(self.seqs)

        return self.sts_tfidf

    def decomposition(self, n_components=16):
        svd = TruncatedSVD(n_components=n_components)
        sts_svd = svd.fit_transform(self.sts_tfidf)

        df_tfidf = pd.DataFrame()
        for i in range(n_components):
            df_tfidf[f'tfidf_{i}'] = sts_svd[:, i]

        return df_tfidf
