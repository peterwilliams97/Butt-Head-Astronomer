from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from framework import LABEL_COLS, df_to_sentences


def get_model(embed_size, maxlen, max_features):
    """Bi-di LSTM with some attention (return_sequences=True)
    """
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


# def df_to_sentences(df):
#     assert not any(df['comment_text'].isnull())
#     return df['comment_text'].fillna('_na_').values


def train_tokenizer(train, max_features):
    sentences = df_to_sentences(train)
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(sentences))
    return tokenizer


def tokenize(tokenizer, df, maxlen):
    sentences = df_to_sentences(df)
    tokens = tokenizer.texts_to_sequences(sentences)
    return sequence.pad_sequences(tokens, maxlen=maxlen)


class ClfKeras:

    def __init__(self, embed_size=128, maxlen=100, max_features=20000):
        self.embed_size = embed_size
        self.maxlen = maxlen
        self.max_features = max_features
        if embed_size == 128 and maxlen == 100 and max_features == 20000:
            self.model_path = 'weights_base.best.hdf5'
        else:
            self.model_path = 'keras_weights_%3d_%3d_%4d.hdf5' % (embed_size, maxlen, max_features)
        self.model = get_model(embed_size, maxlen, max_features)

    def fit(self, train, batch_size=32, epochs=2):
        checkpoint = ModelCheckpoint(self.model_path, monitor='val_loss', verbose=1,
            save_best_only=True, mode='min')
        early = EarlyStopping(monitor='val_loss', mode='min', patience=20)
        callbacks_list = [checkpoint, early]

        self.tokenizer = train_tokenizer(train, self.max_features)
        X_train = tokenize(self.tokenizer, train, maxlen=self.maxlen)
        y_train = train[LABEL_COLS].values
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,
            callbacks=callbacks_list)

    def predict(self, test):
        self.model.load_weights(self.model_path)

        X_test = tokenize(self.tokenizer, test, maxlen=self.maxlen)
        y_test = self.model.predict(X_test)
        return y_test
