import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.callbacks import Callback, EarlyStopping
from utils import dim, xprint, load_model, save_model
from ft_embedding import get_embeddings, get_char_embeddings, tokenize, char_tokenize


model_path = 'gru_rwc_model.pkl'
config_path = 'gru_rwc_model.json'


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.X_char_val, self.y_val = validation_data
        self.best_epoch = -1
        self.best_auc = 0.0

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval != 0:
            return

        y_pred = self.model.predict(x={'w_inp': self.X_val, 'c_inp': self.X_char_val},
            batch_size=2000, verbose=0)

        auc = roc_auc_score(self.y_val, y_pred)
        xprint("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, auc))
        logs['val_auc'] = auc

        if auc >= self.best_auc:
            xprint('RocAucEvaluation.fit: auc=%.3f > best_auc=%.3f' % (auc, self.best_auc))
            self.best_auc = auc
            self.best_epoch = epoch
            save_model(self.model, model_path, config_path)
        else:
            xprint('RocAucEvaluation.fit: No improvement')
        xprint('best_epoch=%d best_auc=%.3f' % (self.best_epoch + 1, self.best_auc))


def get_model(embedding_matrix, char_embedding_matrix,
    max_features, maxlen, char_maxlen, Rec, dropout=0.2, n_hidden=80):

    print('embedding_matrix=%s char_embedding_matrix=%s' % (type(embedding_matrix),
          type(char_embedding_matrix)))
    print('max_features=%s maxlen=%s char_maxlen=%s' % (type(max_features),
          type(maxlen), type(char_maxlen)))
    assert isinstance(max_features, int)
    assert isinstance(maxlen, int)
    assert isinstance(char_maxlen, int)

    inp1 = Input(shape=(maxlen, ), name='w_inp')
    x = Embedding(embedding_matrix.shape[0],
                  embedding_matrix.shape[1],
                  weights=[embedding_matrix], name='w_emb')(inp1)
    x = SpatialDropout1D(dropout, name='w_drop')(x)

    x = Bidirectional(Rec(n_hidden, return_sequences=True), name='w_bidi')(x)
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
    conc2 = Bidirectional(Rec(n_hidden, return_sequences=False), name='c_bidi')(x)

    conc = concatenate([conc1, conc2], name='w_c_conc')

    outp = Dense(6, activation="sigmoid", name='output')(conc)

    model = Model(inputs=[inp1, inp2], outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary(print_fn=xprint)

    return model


class ClfRecWordChar():

    def __init__(self, max_features=30000, char_max_features=1000, maxlen=100,
        dropout=0.2, n_hidden=80, batch_size=32, Rec=None,
        epochs=2, validate=True):
        self.maxlen = maxlen
        self.char_maxlen = maxlen * 4
        self.max_features = max_features
        self.char_max_features = char_max_features
        self.dropout = dropout
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.epochs = epochs
        self.validate = validate
        self.rec = Rec.__name__

        D = self.__dict__
        keys = sorted(D, key=lambda k: (k.startswith('char_'), k))
        self.description = ', '.join('%s=%s' % (k, D[k]) for k in keys)

        self.Rec = Rec
        self.model = None

    def __repr__(self):
        return 'ClfRecWordChar(%s)' % self.description

    def _load_model(self):
        if self.model is not None:
            return
        self.embedding_matrix = get_embeddings(self.max_features, self.maxlen)
        self.char_embedding_matrix = get_char_embeddings(self.char_max_features, self.char_maxlen)

        self.model = get_model(self.embedding_matrix, self.char_embedding_matrix,
              self.max_features, self.maxlen, self.maxlen * 4, self.Rec,
              dropout=self.dropout, n_hidden=self.n_hidden)

    def fit(self, X_train_in, y_train):
        self._load_model()
        X_train = tokenize(self.max_features, self.maxlen, X_train_in)
        X_char_train = char_tokenize(self.char_max_features, self.maxlen, self.char_maxlen, X_train_in)

        print('X_char_train=%s' % dim(X_char_train))
        print('X_char_train=%d %d' % (X_char_train.min(), X_char_train.max()))
        assert np.all(X_char_train <= self.max_features)

        if self.validate:
            X_train, X_val, X_char_train, X_char_val, y_train, y_val = train_test_split(
                X_train, X_char_train, y_train, train_size=0.95, random_state=233)

            RocAuc = RocAucEvaluation(validation_data=(X_val, X_char_val, y_val), interval=1)
            early = EarlyStopping(monitor='val_auc', mode='max', patience=2, verbose=1)
            callback_list = [RocAuc, early]

            self.model.fit(x={"w_inp": X_train, "c_inp": X_char_train}, y=y_train,
                           validation_data=({"w_inp": X_val, "c_inp": X_char_val}, y_val),
                           epochs=self.epochs, callbacks=callback_list, verbose=1)

            xprint('***Best epoch=%d acu=%.4f' % (RocAuc.best_epoch + 1, RocAuc.best_auc))

    def predict(self, X_test_in):
        self.model = load_model(model_path, config_path)
        X_test = tokenize(self.max_features, self.maxlen, X_test_in)
        X_char_test = char_tokenize(self.max_features, self.maxlen, self.char_maxlen, X_test_in)
        y_pred = self.model.predict(x={'w_inp': X_test, 'c_inp': X_char_test}, batch_size=2000)
        return y_pred
