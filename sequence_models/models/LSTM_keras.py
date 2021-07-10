import keras
from keras.preprocessing import text, sequence
from keras.utils import to_categorical
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import *
import tensorflow as tf
from models.Seq2Vec import Seq2Vec


class SeqLSTM(object):
    def __init__(self, seqs, embed_size=64, max_sequence_length=128):
        self.seqs = seqs
        self.embed_size = embed_size
        self.max_sequence_length = max_sequence_length

    def tokenizer(self, max_nb_words):
        tokenizer = text.Tokenizer(num_words=max_nb_words)
        tokenizer.fit_on_texts(self.seqs)
        self.seqs_token = tokenizer.texts_to_sequences(self.seqs)
        self.seqs_token = sequence.pad_sequences(self.seqs_token, maxlen=self.max_sequence_length)
        self.word_index = tokenizer.word_index   # 计算一共出现了多少个单词，其实MAX_NB_WORDS可以直接使用这个数据
        self.nb_words = len(self.word_index) + 1
        print('Total %s word vectors.' % self.nb_words)
        return self.seqs_token

    def build_pre_emb(self):
        # 构建一个embedding矩阵，之后输入到模型使用
        embedding_matrix = np.zeros((self.nb_words, self.embed_size))
        w2v_model = Seq2Vec(emb_size=self.embed_size, window=5, min_count=1, epochs=10)
        w2v_model = w2v_model.fit(self.seqs)
        for word, i in self.word_index.items():
         try:
             embedding_vector = w2v_model.wv.get_vector(word)
         except KeyError:
             continue
         if embedding_vector is not None:
             embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def LSTM(self, embedding_matrix):
        embedding_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        # 词嵌入（使用预训练的词向量）
        embedder = Embedding(self.nb_words,
                             self.embed_size,
                             input_length=self.max_sequence_length,
                             weights=[embedding_matrix],
                             trainable=False)
        embed = embedder(embedding_input)
        lstm = LSTM(128)(embed)
        flat = BatchNormalization()(lstm)
        drop = Dropout(0.2)(flat)
        main_output = Dense(1, activation='sigmoid')(drop)
        model = Model(inputs=embedding_input, outputs=main_output)
        model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy'])
        return model
