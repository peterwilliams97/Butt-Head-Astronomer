import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from os.path import expanduser, join
from framework import SUBMISSION_DIR


TOKENIZE = False
TRAIN = False
PREDICT = False
MAKE_SUBMISSION = True

if MAKE_SUBMISSION:
    PREDICT = True
if TRAIN or PREDICT:
    TOKENIZE = True

def get_model():
    """Bi-di LSTM with some attention (return_sequences=True)
    """
    embed_size = 128
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


max_features = 20000
maxlen = 100

data_dir = expanduser('~/data/toxic/')
model_path = 'weights_base.best.hdf5'
submission_path = join(SUBMISSION_DIR, 'keras_baseline.csv')

if TOKENIZE:
    train = pd.read_csv(join(data_dir, 'train.csv'))
    test = pd.read_csv(join(data_dir, 'test.csv'))
    train = train.sample(frac=1)

    list_sentences_train = train['comment_text'].fillna('CVxTz').values
    list_classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    y = train[list_classes].values
    list_sentences_test = test['comment_text'].fillna('CVxTz').values

    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_train = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

if TRAIN:
    model = get_model()
    batch_size = 32
    epochs = 2

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor='val_loss', mode='min', patience=20)
    callbacks_list = [checkpoint, early]
    model.fit(X_train, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)

if PREDICT:
    model = get_model()
    model.load_weights(model_path)
    y_test = model.predict(X_test)

if MAKE_SUBMISSION:
    submission = pd.read_csv(join(data_dir, 'sample_submission.csv'))
    submission[list_classes] = y_test
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(submission_path, index=False)

