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
from os.path import expanduser, join
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
from framework import MODEL_DIR, LABEL_COLS, df_to_sentences


GLOVE_SIZES = (50, 100, 200, 300)
# Glove dimension 50
EMBEDDING_DIR = expanduser('~/data/glove.6B')


def get_glove_path(embed_size):
    assert embed_size in GLOVE_SIZES, (embed_size, GLOVE_SIZES)
    return join(EMBEDDING_DIR, 'glove.6B.%dd.txt' % embed_size)


if True:
    for embed_size in GLOVE_SIZES:
        embeddings_path = get_glove_path(embed_size)
        assert os.path.exists(embeddings_path), embeddings_path


def get_embeddings_index(embed_size):
    embeddings_path = get_glove_path(embed_size)
    with open(embeddings_path) as f:
        embeddings_index = dict(get_coefs(*o.strip().split()) for o in f)
    return embeddings_index
    # embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_PATH))


def get_coefs(word, *arr):
    """Read the glove word vectors (space delimited strings) into a dictionary from word->vector.
    """
    return word, np.asarray(arr, dtype='float32')


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


def get_embeddings(tokenizer, embed_size, max_features):
    """
        Returns: embedding matrix n_words x embed_size
    """
    # embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_PATH))
    embeddings_index = get_embeddings_index(embed_size)

    # Use these vectors to create our embedding matrix, with random initialization for words
    # that aren't in GloVe. We'll use the same mean and stdev of embeddings the GloVe has when
    # generating the random init.
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    print('emb_mean,emb_std:', emb_mean, emb_std)

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


def get_model(tokenizer, embed_size, maxlen, max_features, dropout=0.1):
    """Bi-di LSTM with some attention (return_sequences=True)
    """
    embedding_matrix = get_embeddings(tokenizer, embed_size, max_features)
    # Bidirectional LSTM with half-size embedding with two fully connected layers
    print('maxlen, [max_features, embed_size], tembedding_matrix', maxlen, [max_features, embed_size],
          embedding_matrix.shape)
    print('embedding_matrix', embedding_matrix.shape)

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


class ClfLstmGlove:

    def __init__(self, embed_size=50, maxlen=100, max_features=20000, dropout=0.1, epochs=3, batch_size=64,
        learning_rate=[0.002, 0.003, 0.000]):
        """
            embed_size: Size of embedding vectors
            maxlen: Max length of comment text
            max_features: Maximum vocabulary size
        """
        self.embed_size = embed_size
        self.maxlen = maxlen
        self.max_features = max_features
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        os.makedirs(MODEL_DIR, exist_ok=True)
        model_name = 'lstm_glove_weights_%03d_%03d_%04d.hdf5' % (embed_size, maxlen, max_features)
        self.model_path = os.path.join(MODEL_DIR, model_name)

        self.description = ', '.join('%s=%s' % (k, v) for k, v in sorted(self.__dict__.items()))

    def __repr__(self):
        return 'ClfLstmGlove(%s)' % self.description

    def fit(self, train):
        self.tokenizer = train_tokenizer(train, self.max_features)
        self.model = get_model(self.tokenizer, self.embed_size, self.maxlen, self.max_features, self.dropout)

        def schedule(epoch):
            n = epoch // len(self.learning_rate)
            m = epoch % len(self.learning_rate)
            fac = 0.5 ** n
            lr = self.learning_rate[m] * fac
            print('^^^ epoch=%d n=%d m=%d fac=%.3f lr=%.5f' % (epoch, n, m, fac, lr))
            return lr

        y_train = train[LABEL_COLS].values
        X_train = tokenize(self.tokenizer, train, maxlen=self.maxlen)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05)
        checkpoint = ModelCheckpoint(self.model_path, monitor='val_auc', verbose=1,
            save_best_only=True, mode='max')
        early = EarlyStopping(monitor='val_auc', mode='max', patience=len(self.learning_rate))
        ra_val = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
        lr = callbacks.LearningRateScheduler(schedule, verbose=1)
        callback_list = [lr, ra_val, checkpoint, early]

        self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs,
                       validation_data=(X_val, y_val), callbacks=callback_list)

    def predict(self, test):
        print('loading model weights %s' % self.model_path)
        self.model.load_weights(self.model_path)
        X_test = tokenize(self.tokenizer, test, maxlen=self.maxlen)
        y_test = self.model.predict([X_test], batch_size=1024, verbose=1)
        return y_test
