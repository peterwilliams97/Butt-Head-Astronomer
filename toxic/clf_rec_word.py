from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.callbacks import Callback, EarlyStopping
from utils import xprint, save_model, load_model
from ft_embedding import get_embeddings, tokenize


model_path = 'gru_rec_word_model.pkl'
config_path = 'gru_rec_word_model.json'


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.best_epoch = -1
        self.best_auc = 0.0

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval != 0:
            return

        y_pred = self.model.predict(self.X_val, verbose=0)
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


def get_model(embedding_matrix, max_features, maxlen, dropout=0.2, n_hidden=80):
    inp = Input(shape=(maxlen, ))
    x = Embedding(embedding_matrix.shape[0],
                  embedding_matrix.shape[1],
                  weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(dropout)(x)
    x = Bidirectional(GRU(n_hidden, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary(print_fn=xprint)

    return model


class ClfRecWord():

    def __init__(self, max_features=30000, maxlen=100, dropout=0.2, n_hidden=80, batch_size=32,
        Rec=None, epochs=2, validate=True):
        self.maxlen = maxlen
        self.max_features = max_features
        self.dropout = dropout
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.epochs = epochs
        self.validate = validate
        self.rec = Rec.__name__

        D = self.__dict__
        self.description = ', '.join('%s=%s' % (k, v) for k, v in sorted(D.items()))

        self.Rec = Rec
        self.model = None

    def __repr__(self):
        return 'ClfRecWord(%s)' % self.description

    def _load_model(self):
        if self.model is not None:
            return
        self.embedding_matrix = get_embeddings(self.max_features, self.maxlen)
        self.model = get_model(self.embedding_matrix, self.max_features, self.maxlen,
            self.dropout, self.n_hidden)

    def fit(self, X_train_in, y_train):
        self._load_model()
        X_train = tokenize(self.max_features, self.maxlen, X_train_in)
        if self.validate:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.95,
                random_state=233)
            RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
            early = EarlyStopping(monitor='val_auc', mode='max', patience=2, verbose=1)
            callback_list = [RocAuc, early]
            self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs,
                           validation_data=(X_val, y_val),
                           callbacks=callback_list, verbose=1)

            self.best_epoch_ = RocAuc.best_epoch + 1
            self.best_auc_ = RocAuc.best_auc
            xprint('***Best epoch=%d auc=%.4f' % (self.best_epoch_, self.best_auc_))

    def predict(self, X_test_in):
        self.model = load_model(model_path, config_path)
        X_test = tokenize(self.max_features, self.maxlen, X_test_in)
        y_pred = self.model.predict(X_test, batch_size=self.batch_size)
        return y_pred
