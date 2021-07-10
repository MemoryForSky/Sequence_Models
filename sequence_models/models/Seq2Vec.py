import numpy as np
import pandas as pd
from gensim.models import Word2Vec


class Seq2Vec(object):
    """
    Word2Vec encoder for the sequence data.

    Parameters:
    -----------
    seqs: 2d-list
        The sequence data.
    emb_size: int, optional
        Dimensionality of the word vectors.
    window: int, optional
        Maximum distance between the current and predicted word within a sentence.
    epochs: int, optional
        Number of iterations (epochs) over the corpus. (Formerly: iter)
    seed: int, optional
        Seed for the random number generator.
    """
    def __init__(self, emb_size=32, window=6, min_count=5, epochs=5, seed=2021):
        self.emb_size = emb_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.seed = seed
        self.seqs = None

    def fit(self, seqs):
        if isinstance(seqs[0], list):
            for i in range(len(seqs)):
                seqs[i] = [str(x) for x in seqs[i]]
        else:
            raise Exception("Error! cannot deal with the type of seqs.")

        self.seqs = seqs
        model = Word2Vec(self.seqs,
                         vector_size=self.emb_size,
                         window=self.window,
                         min_count=self.min_count,
                         sg=0,
                         hs=0,
                         seed=self.seed,
                         epochs=self.epochs)
        return model

    def get_matrix(self, model):
        emb_matrix = []
        for seq in self.seqs:
            vec = []
            for w in seq:
                if w in model.wv:
                    vec.append(model.wv.get_vector(w))
            if len(vec) > 0:
                emb_matrix.append(np.mean(vec, axis=0))
            else:
                emb_matrix.append([0] * self.emb_size)
        emb_matrix = np.array(emb_matrix)
        df_emb = pd.DataFrame()
        for i in range(self.emb_size):
            df_emb['emb_{}'.format(i)] = emb_matrix[:, i]

        return df_emb
