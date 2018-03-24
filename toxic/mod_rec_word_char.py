import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from utils import dim, xprint
from ft_embedding import get_embeddings, get_char_embeddings, tokenize, char_tokenize


def get_model1(embedding_matrix, char_embedding_matrix,
    max_features, maxlen, char_maxlen, Rec,
    dropout=0.2, rnn_layers=1, n_hidden=80, trainable=True):

    # print('embedding_matrix=%s char_embedding_matrix=%s' % (type(embedding_matrix),
    #       type(char_embedding_matrix)))
    # print('max_features=%s maxlen=%s char_maxlen=%s' % (type(max_features),
    #       type(maxlen), type(char_maxlen)))
    assert isinstance(max_features, int)
    assert isinstance(maxlen, int)
    assert isinstance(char_maxlen, int)

    inp1 = Input(shape=(maxlen, ), name='w_inp')
    x = Embedding(embedding_matrix.shape[0],
                  embedding_matrix.shape[1],
                  weights=[embedding_matrix],
                  trainable=trainable, name='w_emb')(inp1)
    x = SpatialDropout1D(dropout, name='w_drop')(x)

    for i in range(rnn_layers):
        x = Bidirectional(Rec(n_hidden, return_sequences=True), name='w_bidi%d' % i)(x)
    avg_pool = GlobalAveragePooling1D(name='w_ave')(x)
    max_pool = GlobalMaxPooling1D(name='w_max')(x)
    conc1 = concatenate([avg_pool, max_pool], name='w_conc')
    # conc1 = Bidirectional(Rec(n_hidden, return_sequences=False))(x)

    inp2 = Input(shape=(char_maxlen, ), name='c_inp')
    x = Embedding(char_embedding_matrix.shape[0],
                  char_embedding_matrix.shape[1],
                  weights=[char_embedding_matrix], name='c_emb')(inp2)
    x = SpatialDropout1D(dropout, name='c_drop')(x)

    # x = Bidirectional(LSTM(n_hidden, return_sequences=True))(x)
    # avg_pool = GlobalAveragePooling1D()(x)
    # max_pool = GlobalMaxPooling1D()(x)
    # conc2 = concatenate([avg_pool, max_pool])
    for i in range(rnn_layers - 1):
        x = Bidirectional(Rec(n_hidden, return_sequences=True), name='c_bidi%d' % i)(x)
    conc2 = Bidirectional(Rec(n_hidden, return_sequences=False), name='c_bidi%d' % rnn_layers)(x)

    conc = concatenate([conc1, conc2], name='w_c_conc')

    outp = Dense(6, activation="sigmoid", name='output')(conc)

    model = Model(inputs=[inp1, inp2], outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def get_model2(embedding_matrix, char_embedding_matrix,
    max_features, maxlen, char_maxlen, Rec,
    dropout=0.2, rnn_layers=1, n_hidden=80, trainable=True):

    # print('embedding_matrix=%s char_embedding_matrix=%s' % (type(embedding_matrix),
    #       type(char_embedding_matrix)))
    # print('max_features=%s maxlen=%s char_maxlen=%s' % (type(max_features),
    #       type(maxlen), type(char_maxlen)))
    assert isinstance(max_features, int)
    assert isinstance(maxlen, int)
    assert isinstance(char_maxlen, int)

    inp1 = Input(shape=(maxlen, ), name='w_inp')
    x = Embedding(embedding_matrix.shape[0],
                  embedding_matrix.shape[1],
                  weights=[embedding_matrix],
                  trainable=trainable, name='w_emb')(inp1)
    x = SpatialDropout1D(dropout, name='w_drop')(x)

    for i in range(rnn_layers):
        x = Bidirectional(Rec(n_hidden, return_sequences=True), name='w_bidi%d' % i)(x)
    avg_pool = GlobalAveragePooling1D(name='w_ave')(x)
    max_pool = GlobalMaxPooling1D(name='w_max')(x)
    conc1 = concatenate([avg_pool, max_pool], name='w_conc')
    # conc1 = Bidirectional(Rec(n_hidden, return_sequences=False))(x)

    inp2 = Input(shape=(char_maxlen, ), name='c_inp')
    x = Embedding(char_embedding_matrix.shape[0],
                  char_embedding_matrix.shape[1],
                  weights=[char_embedding_matrix], name='c_emb')(inp2)
    x = SpatialDropout1D(dropout, name='c_drop')(x)

    for i in range(rnn_layers):
        x = Bidirectional(Rec(n_hidden, return_sequences=True), name='c_bidi%d' % i)(x)
    avg_pool = GlobalAveragePooling1D(name='c_ave')(x)
    max_pool = GlobalMaxPooling1D(name='c_max')(x)
    conc2 = concatenate([avg_pool, max_pool], name='c_conc')
    # conc2 = Bidirectional(Rec(n_hidden, return_sequences=False), name='c_bidi')(x)

    conc = concatenate([conc1, conc2], name='w_c_conc')

    outp = Dense(6, activation="sigmoid", name='output')(conc)

    model = Model(inputs=[inp1, inp2], outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


type_model = {
    1: get_model1,
    2: get_model2
}


class ClfRecWordChar():

    def __init__(self, max_features=30000, char_max_features=1000, maxlen=100,
        dropout=0.2, n_hidden=80, trainable=True, batch_size=32, Rec=None, model_type=1,
        epochs=1):
        self.maxlen = maxlen
        self.char_maxlen = maxlen * 4
        self.max_features = max_features
        self.char_max_features = char_max_features
        self.dropout = dropout
        self.n_hidden = n_hidden
        self.trainable = trainable
        self.batch_size = batch_size
        self.epochs = epochs
        self.rec = Rec.__name__
        self.model_type = model_type

        D = self.__dict__
        keys = sorted(D, key=lambda k: (k.startswith('char_'), k))
        self.description = ', '.join('%s=%s' % (k, D[k]) for k in keys)

        self.Rec = Rec
        self.model = None
        self.preprocessed = False

    def __repr__(self):
        return 'ClfRecWordChar(%s)' % self.description

    def __del__(self):
        if self.model:
            del self.model

    def show_model(self):
        self._load_model()
        self.model.summary(print_fn=xprint)

    def _load_model(self):
        if self.model is not None:
            return
        self.embedding_matrix = get_embeddings(self.max_features, self.maxlen)
        self.char_embedding_matrix = get_char_embeddings(self.char_max_features, self.char_maxlen)

        get_model = type_model[self.model_type]

        self.model = get_model(self.embedding_matrix, self.char_embedding_matrix,
              self.max_features, self.maxlen, self.maxlen * 4, self.Rec,
              dropout=self.dropout, n_hidden=self.n_hidden, trainable=self.trainable)

    def prepare_fit(self, X_train_in):
        if not self.preprocessed:
            self._load_model()
            X_train = tokenize(self.max_features, self.maxlen, X_train_in)
            X_char_train = char_tokenize(self.char_max_features, self.char_maxlen,
                X_train_in)

            print('char_max_features=%d char_maxlen=%d' % (self.char_max_features, self.char_maxlen))
            print('X_char_train=%s' % dim(X_char_train))
            print('X_char_train=%d %d' % (X_char_train.min(), X_char_train.max()))
            assert np.all(X_char_train <= self.char_max_features)

            self.X_train = X_train
            self.X_char_train = X_char_train
            self.preprocessed = True

        return self.X_train, self.X_char_train

    def fit(self, X_train_in, y_train):
        X_train, X_char_train = self.prepare_fit(X_train_in)
        self.model.fit(x={"w_inp": X_train, "c_inp": X_char_train}, y=y_train,
                       epochs=self.epochs, verbose=1)

    def predict(self, X_test_in):
        X_test = tokenize(self.max_features, self.maxlen, X_test_in)
        X_char_test = char_tokenize(self.char_max_features, self.char_maxlen, X_test_in)
        y_pred = self.model.predict(x={'w_inp': X_test, 'c_inp': X_char_test}, batch_size=2000)
        return y_pred
