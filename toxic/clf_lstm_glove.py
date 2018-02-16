# coding: utf-8
"""
  LSTM + GloVe + Cross-validation + LearningRate changes + ...

    Notes
    1) GRU is very similar to LSTM and not better

    2) GloVe dimension is very important. I recommend to use GloVe 840b 300d if you can (it's very
    hard to use it in kaggle kernels)

    3) Cross Validation is interesting for hiperparameters tuning, but for higher score you shoudn't
     use validation_split

    4) First Epoch is very unstable. So I use small LR on first step

    5) Dataset size is small. So you may use some additional datasets and then finetune model

    6) It's hard not to overfit the model and I have n't found yet a good way to solve this problem.
    BatchNormalization/Dropout don't really help.
    https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/46494263247
    Maybe Reccurent Batch Normalization can help, but it is'not implemented in keras.

    7) Use Attention layer from here (AttLayer):
    https://github.com/dem-esgal/textClassifier/blob/master/textClassifierHATT.py

    Thanks to (https://www.kaggle.com/CVxTz/keras-bidirectional-lstm-baseline-lb-0-051)
"""
import os
import time
from os.path import join
import numpy as np
from keras.preprocessing import text
from keras.preprocessing import sequence
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from functools import partial
from framework import MODEL_DIR, LABEL_COLS, df_to_sentences
from utils import DATA_ROOT, dim


GLOVE_SETS = {
    'twitter': ('glove.twitter.27B', tuple([25, 50, 100, 200])),
    '6B': ('glove.6B', tuple([50, 100, 200, 300])),
    # '840B': ('glove.840B.300d', tuple([300]))
}

GLOVE_SIZES = (50, 100, 200, 300)
# Glove dimension 50
EMBEDDING_DIR = join(DATA_ROOT, 'glove.6B')


def get_embedding_path(embed_name, embed_size):
    glove_name, glove_sizes = GLOVE_SETS[embed_name]
    assert embed_size in glove_sizes, (embed_name, embed_size, glove_sizes)
    embedding_dir = join(DATA_ROOT, glove_name)
    assert os.path.exists(embedding_dir), embedding_dir
    embedding_path = join(embedding_dir, '%s.%dd.txt' % (glove_name, embed_size))
    assert os.path.exists(embedding_path), embedding_path
    return embedding_path


if True:
    for embed_name, (glove_name, glove_sizes) in GLOVE_SETS.items():
        for embed_size in glove_sizes:
            embeddings_path = get_embedding_path(embed_name, embed_size)
            assert os.path.exists(embeddings_path), embeddings_path


def get_embeddings_index(embed_name, embed_size):
    assert embed_name in GLOVE_SETS, embed_name
    embeddings_path = get_embedding_path(embed_name, embed_size)
    embeddings_index = {}
    with open(embeddings_path, 'rb') as f:
        t0 = time.clock()
        for i, line in enumerate(f):
            parts = line.strip().split()
            embeddings_index[parts[0]] = np.asarray(parts[1:], dtype='float32')
            if (i + 1) % 200000 == 0:
                print('%7d embeddings %.1f sec' % (i + 1, time.clock() - t0))
    print('%7d embeddings %.1f sec' % (len(embeddings_index), time.clock() - t0))
    return embeddings_index


class RocAucEvaluation(Callback):
    """ROC AUC for CV in Keras see for details: https://gist.github.com/smly/d29d079100f8d81b905e
    """

    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print('\n ROC-AUC - epoch: {:d} - score: {:.6f}'.format(epoch, score))
            logs['val_auc'] = score


def get_embeddings(tokenizer, embed_name, embed_size, max_features):
    """Returns: embedding matrix n_words x embed_size
    """
    assert embed_name in GLOVE_SETS, embed_name
    embeddings_index = get_embeddings_index(embed_name, embed_size)

    # Use these vectors to create our embedding matrix, with random initialization for words
    # that aren't in GloVe. We'll use the same mean and stdev of embeddings the GloVe has when
    # generating the random init.
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    print('emb_mean=%.3f emb_std=%.3f' % (emb_mean, emb_std))

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def get_model(tokenizer, embed_name, embed_size, maxlen, max_features, dropout):
    """Bi-di LSTM with some attention (return_sequences=True)
    """
    assert embed_name in GLOVE_SETS, embed_name
    embedding_matrix = get_embeddings(tokenizer, embed_name, embed_size, max_features)
    # Bidirectional LSTM with half-size embedding with two fully connected layers
    print('maxlen=%d [max_features, embed_size]=%s, embedding_matrix%s' % (maxlen,
        [max_features, embed_size], dim(embedding_matrix)))
    print('embedding_matrix', dim(embedding_matrix))

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True, dropout=dropout, recurrent_dropout=dropout))(x)
    x = GlobalMaxPool1D()(x)
    x = BatchNormalization()(x)
    x = Dense(50, activation="relu")(x)
    #x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    def loss(y_true, y_pred):
        return K.binary_crossentropy(y_true, y_pred)

    # Add AUC to metrics !@#$
    model.compile(loss=loss, optimizer='nadam', metrics=['accuracy'])

    return model


def train_tokenizer(train, max_features):
    sentences = df_to_sentences(train)
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(sentences))
    return tokenizer


def tokenize(tokenizer, df, maxlen):
    sentences = df_to_sentences(df)
    tokens = tokenizer.texts_to_sequences(sentences)
    return sequence.pad_sequences(tokens, maxlen=maxlen)


def lr_schedule(epoch, learning_rate):
    m = min(len(learning_rate) - 1, epoch)
    return learning_rate[m]
    n = epoch // len(learning_rate)
    m = epoch % len(learning_rate)
    fac = 0.5 ** n
    lr = learning_rate[m] * fac
    print('^^^ epoch=%d n=%d m=%d fac=%.3f lr=%.5f' % (epoch, n, m, fac, lr))
    return lr


def get_model_path(model_name, fold):
    return os.path.join(MODEL_DIR, '%s_%d.hdf5' % (model_name, fold))


n_folds = 2


class ClfLstmGlove:

    def __init__(self, embed_name='6B', embed_size=50, maxlen=100, max_features=20000, dropout=0.1,
        epochs=3, batch_size=64, learning_rate=[0.002, 0.003, 0.000]):
        """
            embed_size: Size of embedding vectors
            maxlen: Max length of comment text
            max_features: Maximum vocabulary size
        """
        assert embed_name in GLOVE_SETS, embed_name
        self.embed_name = embed_name
        self.embed_size = embed_size
        self.maxlen = maxlen
        self.max_features = max_features
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        os.makedirs(MODEL_DIR, exist_ok=True)
        self.model_name = 'lstm_glove_weights_%03d_%03d_%04d' % (embed_size, maxlen, max_features)

        self.description = ', '.join('%s=%s' % (k, v) for k, v in sorted(self.__dict__.items()))

    def __repr__(self):
        return 'ClfLstmGlove(%s)' % self.description

    def fit(self, train):
        self.tokenizer = train_tokenizer(train, self.max_features)
        self.model = get_model(self.tokenizer, self.embed_name, self.embed_size, self.maxlen,
            self.max_features, self.dropout)

        y_train0 = train[LABEL_COLS].values
        X_train0 = tokenize(self.tokenizer, train, maxlen=self.maxlen)

        for fold in range(n_folds):
            X_train, X_val, y_train, y_val = train_test_split(X_train0, y_train0, test_size=0.1)
            self.fit_fold(X_train, X_val, y_train, y_val, fold=fold)

    def fit_fold(self, X_train, X_val, y_train, y_val, fold):
        print('fitting %d of %d folds X_train=%s X_val=%s' % (fold, n_folds, dim(X_train), dim(X_val)))
        model_path = get_model_path(self.model_name, fold)

        checkpoint = ModelCheckpoint(model_path, monitor='val_auc', verbose=1,
            save_best_only=True, mode='max')
        early = EarlyStopping(monitor='val_auc', mode='max', patience=len(self.learning_rate))
        ra_val = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
        schedule = partial(lr_schedule, learning_rate=self.learning_rate)
        lr = callbacks.LearningRateScheduler(schedule, verbose=1)
        callback_list = [lr, ra_val, checkpoint, early]

        self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs,
                       validation_data=(X_val, y_val), callbacks=callback_list)

    def predict(self, test):
        X_test = tokenize(self.tokenizer, test, maxlen=self.maxlen)
        y_test_folds = [self.predict_fold(X_test, fold=fold) for fold in range(n_folds)]
        y0 = y_test_folds[0]
        y_test = np.zeros((n_folds, y0.shape[0], y0.shape[1]), dtype=y0.dtype)
        for fold in range(n_folds):
            y_test[fold, :, :] = y_test_folds[fold]
        return y_test.mean(axis=0)

    def predict_fold(self, X_test, fold):
        model_path = get_model_path(self.model_name, fold)
        print('loading model weights %s' % model_path)
        self.model.load_weights(model_path)
        y_test = self.model.predict([X_test], batch_size=1024, verbose=1)
        return y_test
