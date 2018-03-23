from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from utils import xprint
from ft_embedding import get_embeddings, tokenize


def get_model(embedding_matrix, max_features, maxlen, Rec, dropout=0.2, n_hidden=80, trainable=True):
    inp = Input(shape=(maxlen, ))
    x = Embedding(embedding_matrix.shape[0],
                  embedding_matrix.shape[1],
                  weights=[embedding_matrix], trainable=trainable)(inp)
    x = SpatialDropout1D(dropout)(x)
    x = Bidirectional(Rec(n_hidden, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


class ClfRecWord():

    def __init__(self, max_features=30000, maxlen=100, dropout=0.2, n_hidden=80, batch_size=32,
        trainable=True, Rec=None, epochs=2):
        self.maxlen = maxlen
        self.max_features = max_features
        self.dropout = dropout
        self.n_hidden = n_hidden
        self.trainable = trainable
        self.batch_size = batch_size
        self.epochs = epochs
        self.rec = Rec.__name__

        D = self.__dict__
        keys = sorted(D, key=lambda k: (k.startswith('char_'), k))
        self.description = ', '.join('%s=%s' % (k, D[k]) for k in keys)

        self.Rec = Rec
        self.model = None
        self.preprocessed = False

    def __repr__(self):
        return 'ClfRecWordChar(%s)' % self.description

    def show_model(self):
        self._load_model()
        self.model.summary(print_fn=xprint)

    def _load_model(self):
        if self.model is not None:
            return
        self.embedding_matrix = get_embeddings(self.max_features, self.maxlen)
        self.model = get_model(self.embedding_matrix, self.max_features, self.maxlen, self.Rec,
              dropout=self.dropout, n_hidden=self.n_hidden, trainable=self.trainable)

    def prepare_fit(self, X_train_in):
        if not self.preprocessed:
            self._load_model()
            self.X_train = tokenize(self.max_features, self.maxlen, X_train_in)
            self.preprocessed = True
        return self.X_train

    def fit(self, X_train_in, y_train):
        X_train = self.prepare_fit(X_train_in)
        self.model.fit(x=X_train, y=y_train, epochs=self.epochs, verbose=1)

    def predict(self, X_test_in):
        X_test = tokenize(self.max_features, self.maxlen, X_test_in)
        y_pred = self.model.predict(x=X_test, batch_size=2000)
        return y_pred
