import cytoolz
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import (LSTM, Dense, Embedding, Bidirectional, Dropout, GlobalMaxPool1D,
    GlobalAveragePooling1D, BatchNormalization, TimeDistributed, Flatten)
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from spacy.compat import pickle
import spacy
import os
import time
import math
import multiprocessing
from framework import MODEL_DIR, LABEL_COLS, N_SAMPLES, df_to_sentences, train_test_split
from utils import dim, xprint, RocAucEvaluation, Cycler, save_model
from spacy_glue import SpacySentenceWordCache


n_threads = max(multiprocessing.cpu_count() - 1, 1)
xprint('n_threads=%d' % n_threads)
sentence_cache = SpacySentenceWordCache()


if False:
    for lang in ('en', 'en_vectors_web_lg', 'en_core_web_lg'):
        nlp = spacy.load(lang)
        print('lang=%-16s pipe_names=%s' % (lang, nlp.pipe_names))
    assert False


MIN, MEAN, MAX, MEAN_MAX, MEDIAN, PC75, PC90, LINEAR, LINEAR2, LINEAR3, LINEAR4, LINEAR5, EXP = (
    'MIN', 'MEAN', 'MAX', 'MEAN_MAX',
    'MEDIAN', 'PC75',
    'PC90', 'LINEAR', 'LINEAR2', 'LINEAR3', 'LINEAR4', 'LINEAR5', 'EXP')
PREDICT_METHODS = (MIN, MEAN, MAX, MEAN_MAX, MEDIAN, PC75, PC90, LINEAR, LINEAR2, LINEAR3, LINEAR4,
    LINEAR5, EXP)


def linear_weights(ys, limit):
    n = ys.shape[0]
    weights = np.ones(n, dtype=np.float64)
    if n <= 1:
        return weights
    lo = limit
    hi = 1.0 - limit
    span = hi - lo
    d = span / (n - 1)
    for i in range(n):
        weights[i] = lo + d * i
    weights /= weights.sum()
    return weights


def exponential_weights(ys, limit):
    n = ys.shape[0]
    weights = np.ones(n, dtype=np.float64)
    if n <= 1:
        return weights
    lo = limit
    hi = 1.0 - limit
    d = hi / lo
    assert d > 1.0
    for i in range(n):
        weights[i] = lo * (d ** i)
    weights /= weights.sum()
    for i in range(n):
        assert 0.0 < weights[i] < 1.0
    return weights


def reduce(ys_in, method):
    ys = ys_in.copy()
    for j in range(ys.shape[1]):
        ys[:, j] = np.sort(ys_in[:, j])
    # print(ys_in)
    # print(ys)
    # assert False
    if method == MIN:
        return ys.min(axis=0)
    if method == MEAN:
        return ys.mean(axis=0)
    elif method == MAX:
        return ys.max(axis=0)
    elif method == MEAN_MAX:
        return (ys.mean(axis=0) + ys.max(axis=0)) / 2.0
    elif method == MEDIAN:
        return np.percentile(ys, 50.0, axis=0, interpolation='higher')
    elif method == PC75:
        return np.percentile(ys, 75.0, axis=0, interpolation='higher')
    elif method == PC90:
        return np.percentile(ys, 90.0, axis=0, interpolation='higher')
    elif method == LINEAR:
        weights = linear_weights(ys, limit=0.1)
        y = np.dot(weights, ys)
        assert len(y.shape) == 1 and len(y) == ys.shape[1], (weights.shape, ys.shape, y.shape)
        return y
    elif method == LINEAR2:
        weights = linear_weights(ys, limit=0.2)
        return np.dot(weights, ys)
    elif method == LINEAR3:
        weights = linear_weights(ys, limit=0.3)
        return np.dot(weights, ys)
    elif method == LINEAR4:
        weights = linear_weights(ys, limit=0.05)
        return np.dot(weights, ys)
    elif method == LINEAR5:
        weights = linear_weights(ys, limit=0.01)
        return np.dot(weights, ys)
    elif method == EXP:
        weights = exponential_weights(ys, limit=0.3)
        y = np.dot(weights, ys)
        assert len(y.shape) == 1 and len(y) == ys.shape[1], (weights.shape, ys.shape, y.shape)
        return y
    raise ValueError('Bad method=%s' % method)


if False:

    def test_reduce(ys):
        print('-' * 80)
        print(ys)
        for method in PREDICT_METHODS:
            y = reduce(ys, method)
            print('%8s: %s' % (method, y))

    ys = np.ones((1, 4), dtype=np.float32)
    test_reduce(ys)

    ys = np.ones((6, 4), dtype=np.float32)
    ys[:, 0] = 0.0
    ys[:3, 2] = -1.0
    ys[:3, 1] = 2.0
    test_reduce(ys)
    assert False


class SentimentAnalyser(object):
    @classmethod
    def load(cls, path, nlp, method, max_length, frozen):
        xprint('SentimentAnalyser.load: path=%s max_length=%d' % (path, max_length))
        with open(os.path.join(path, 'config.json'), 'rt') as f:
            model = model_from_json(f.read())
        with open(os.path.join(path, 'model'), 'rb') as f:
            lstm_weights = pickle.load(f)
        if frozen:
            embeddings = get_embeddings(nlp.vocab)
            lstm_weights = [embeddings] + lstm_weights
        model.set_weights(lstm_weights)
        return cls(model, method=method, max_length=max_length)

    def __init__(self, model, method, max_length):
        self._model = model
        self.method = method
        self.max_length = max_length

    def pipe(self, docs, batch_size=1000, n_threads=n_threads):
        for minibatch in cytoolz.partition_all(batch_size, docs):
            minibatch = list(minibatch)
            for doc in minibatch:
                Xs = get_features(doc.sents, self.max_length)
                ys = self._model.predict(Xs)
                y = reduce(ys, method=self.method)
                assert len(y.shape) == 1 and len(y) == ys.shape[1], (weights.shape, ys.shape, y.shape)
                doc.user_data['toxics'] = y
                yield doc


def sentence_label_generator(texts_in, labels_in, batch_size):
    print('sentence_label_generator', len(texts_in), len(labels_in), batch_size)
    sent_labels = []

    texts_cycler = Cycler(texts_in, batch_size)
    labels_cycler = Cycler(labels_in, batch_size)

    while True:
        texts = texts_cycler.batch()
        labels = labels_cycler.batch()
        text_sents = sentence_cache.sent_id_pipe(texts)
        for sents, label in zip(text_sents, labels):
            for sent in sents:
                 sent_labels.append((sent, label))
        while len(sent_labels) >= batch_size:
            yield sent_labels[:batch_size]
            sent_labels = sent_labels[batch_size:]


def sentence_feature_generator(max_length, batch_size, texts_in, labels_in, name, n_sentences):
    t0 = time.clock()
    N = max(1, n_sentences / 5)
    n = 0

    gen = sentence_label_generator(texts_in, labels_in, batch_size)

    while True:
        sent_labels = next(gen)
        Xs = np.zeros((batch_size, max_length), dtype='int32')
        ys = np.zeros((batch_size, len(LABEL_COLS)), dtype='int32')
        for i, (sent, y) in enumerate(sent_labels):
            for j, vector_id in enumerate(sent[:max_length]):
                if vector_id >= 0:
                    Xs[i, j] = vector_id
            ys[i, :] = y
        # print('#### Xs=%s ys=%s' % (dim(Xs), dim(ys)))
        yield Xs, ys
        n += batch_size
        if n % N == 0:
            dt = max(time.clock() - t0, 1.0)
            print('^^^^%5s %7d (%5.1f%%) sents dt=%4.1f sec %3.1f sents/sec' %
                (name, n, 100.0 * n / n_sentences, dt, n / dt))


def make_sentences(max_length, batch_size, texts_in, labels_in, name, n_sentences):
    t0 = time.clock()
    N = max(1, n_sentences / 5)
    n = 0

    Xs = np.zeros((n_sentences, max_length), dtype='int32')
    ys = np.zeros((n_sentences, len(LABEL_COLS)), dtype='int32')
    print('make_sentences: Xs=%s ys=%s' % (dim(Xs), dim(ys)))

    gen = sentence_label_generator(texts_in, labels_in, batch_size)
    while n < n_sentences:
        sent_labels = next(gen)
        m = min(batch_size, n_sentences - n)
        # assert n + m <= n_sentences, (n, m, n_sentences)
        for i, (sent, y) in enumerate(sent_labels[:m]):
            # assert n + i < n_sentences, (n, i, n_sentences)
            for j, vector_id in enumerate(sent[:max_length]):
                if vector_id >= 0:
                    Xs[n + i, j] = vector_id
            ys[n + i, :] = y
        n += m
        if n % N == 0 or n + 1 == n_sentences:
            dt = max(time.clock() - t0, 1.0)
            print('^^^^%5s %7d (%5.1f%%) sents dt=%4.1f sec %3.1f sents/sec' %
                (name, n, 100.0 * n / n_sentences, dt, n / dt))

    return Xs, ys


def count_sentences(texts_in, batch_size, name):
    t0 = time.clock()
    N = max(1, len(texts_in) / 5)
    n = 0
    k = 0
    for minibatch in cytoolz.partition_all(batch_size, texts_in):
        texts = list(minibatch)
        text_sents = sentence_cache.sent_id_pipe(texts)
        k += len(text_sents)
        n += sum(len(sent) for sent in text_sents)
        if k % N == 0 or k == len(texts):
            dt = max(time.clock() - t0, 1.0)
            print('##^^%5s=%7d sents=%8d dt=%4.1f sec %2.1f sents/doc %3.1f docs/sec %3.1f sents/sec' %
                (name, k, n, dt, n / k, k / dt, n / dt))
    return n


def do_train(train_texts, train_labels, dev_texts, dev_labels,
    lstm_shape, lstm_settings, lstm_optimizer, batch_size=100, epochs=5, by_sentence=True,
    frozen=False, lstm_type=1, model_path=None):
    """Train a Keras model on the sentences in `train_texts`
        All the sentences in a text have the text's label
    """

    print('do_train: train_texts=%s dev_texts=%s' % (dim(train_texts), dim(dev_texts)))
    best_epoch_frozen, best_epoch_unfrozen = -1, -1

    n_train_sents = count_sentences(train_texts, batch_size, 'train')
    X_train, y_train = make_sentences(lstm_shape['max_length'], batch_size,
        train_texts, train_labels, 'train', n_train_sents)
    validation_data = None
    if dev_texts is not None:
        n_dev_sents = count_sentences(dev_texts, batch_size, 'dev')
        X_val, y_val = make_sentences(lstm_shape['max_length'], batch_size,
            dev_texts, dev_labels, 'dev', n_dev_sents)
        validation_data = (X_val, y_val)
    sentence_cache.flush()

    print("Loading spaCy")
    nlp = sentence_cache._load_nlp()
    embeddings = get_embeddings(nlp.vocab)
    model = build_lstm[lstm_type](embeddings, lstm_shape, lstm_settings)
    compile_lstm(model, lstm_settings['lr'])

    callback_list = None
    if validation_data is not None:
        ra_val = RocAucEvaluation(validation_data=validation_data, interval=1, frozen=frozen,
            model_path=model_path)
        early = EarlyStopping(monitor='val_auc', mode='max', patience=2, verbose=1)
        callback_list = [ra_val, early]

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=validation_data, callbacks=callback_list, verbose=1)
    if validation_data is not None:
        best_epoch_frozen = ra_val.best_epoch
        ra_val.best_epoch = -1
    else:
        save_model(model, frozen, model_path)

    if not frozen:
        xprint("Unfreezing")
        for layer in model.layers:
            layer.trainable = True
        compile_lstm(model, lstm_settings['lr'] / 10)
        if validation_data is not None:
            # Reload the best model so far
            lstm_weights = [embeddings] + ra_val.top_weights
            model.set_weights(lstm_weights)
            # Reset early stopping
            early = EarlyStopping(monitor='val_auc', mode='max', patience=2, verbose=1)
            callback_list = [ra_val, early]
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=validation_data, callbacks=callback_list, verbose=1)
        best_epoch_unfrozen = ra_val.best_epoch

    return model, (best_epoch_frozen, best_epoch_unfrozen)


def get_features(docs, max_length):
    docs = list(docs)
    Xs = np.zeros((len(docs), max_length), dtype='int32')
    for i, doc in enumerate(docs):
        for j, token in enumerate(doc):
            if j >= max_length:
                break
            vector_id = token.vocab.vectors.find(key=token.orth)
            if vector_id >= 0:
                Xs[i, j] = vector_id
    return Xs


def build_lstm1(embeddings, shape, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=True
        )
    )
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False)))
    model.add(Bidirectional(LSTM(shape['n_hidden'],
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout'])))
    model.add(Dense(shape['n_class'], activation='sigmoid'))
    xprint('build_lstm1: embeddings=%s shape=%s' % (dim(embeddings), shape))
    return model


def build_lstm2(embeddings, shape, settings):
    # inp = Input(shape=(shape['max_length'],))
    # x = Embedding(
    #         embeddings.shape[0],
    #         embeddings.shape[1],
    #         input_length=shape['max_length'],
    #         trainable=False,
    #         weights=[embeddings],
    #         mask_zero=True
    #     )(inp)
    # x = Bidirectional(LSTM(shape['n_hidden'],
    #                              recurrent_dropout=settings['dropout'],
    #                              dropout=settings['dropout']))(x)
    # x = GlobalMaxPool1D()(x)
    # x = BatchNormalization()(x)
    # x = Dense(50, activation="relu")(x)
    # #x = BatchNormalization()(x)
    # x = Dropout(dropout)(x)
    # x = Dense(shape['n_class'], activation='sigmoid')(x)
    # model = Model(inputs=inp, outputs=x)

    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=False
        )
    )
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False)))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout'])))
    model.add(GlobalMaxPool1D())
    model.add(BatchNormalization())
    model.add(Dense(shape['n_class'], activation='sigmoid'))
    xprint('build_lstm2: embeddings=%s shape=%s' % (dim(embeddings), shape))
    return model


def build_lstm3(embeddings, shape, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=False,
            name='eembed'
        )
    )
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False, name='td')))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout'])))
    model.add(Flatten(name='flaaten'))
    model.add(BatchNormalization())
    model.add(Dense(shape['n_class'], activation='sigmoid'))
    xprint('build_lstm3: embeddings=%s shape=%s' % (dim(embeddings), shape))
    return model


def build_lstm4(embeddings, shape, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=False,
            name='eembed'
        )
    )
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False, name='td')))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout'])))
    model.add(Flatten(name='flaaten'))
    model.add(BatchNormalization())
    n_dense = int(math.ceil(math.sqrt(shape['n_hidden'] * shape['n_class'])))
    model.add(Dense(n_dense, activation='relu'))
    # model.add(BatchNormalization())
    # x = Dropout(dropout)(x)
    model.add(Dense(shape['n_class'], activation='sigmoid'))
    xprint('build_lstm4: embeddings=%s shape=%s' % (dim(embeddings), shape))
    return model


def build_lstm5(embeddings, shape, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=False
        )
    )
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False)))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout'])))
    model.add(GlobalAveragePooling1D())
    model.add(BatchNormalization())
    model.add(Dense(shape['n_class'], activation='sigmoid'))
    xprint('build_lstm5: embeddings=%s shape=%s' % (dim(embeddings), shape))
    return model


def build_lstm6(embeddings, shape, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=False
        )
    )
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False)))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout'])))
    model.add(GlobalMaxPool1D())
    model.add(BatchNormalization())
    model.add(Dropout(settings['dropout'] / 2.0))
    model.add(Dense(shape['n_class'], activation='sigmoid'))
    xprint('build_lstm6: embeddings=%s shape=%s' % (dim(embeddings), shape))
    return model


def build_lstm7(embeddings, shape, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=False
        )
    )
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False)))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout'])))
    model.add(GlobalMaxPool1D())
    model.add(BatchNormalization())
    # model.add(Dropout(settings['dropout'] / 2.0))
    model.add(Dense(shape['n_hidden'] // 2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(settings['dropout'] / 2.0))
    model.add(Dense(shape['n_class'], activation='sigmoid'))
    xprint('build_lstm7: embeddings=%s shape=%s' % (dim(embeddings), shape))
    return model


def build_lstm8(embeddings, shape, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=False,
            name='eembed'
        )
    )
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False, name='td')))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout']), name='bidi'))
    model.add(Flatten(name='flaaten'))
    model.add(BatchNormalization())
    model.add(Dropout(settings['dropout'] / 2.0))
    model.add(Dense(shape['n_class'], activation='sigmoid'))
    xprint('build_lstm8: embeddings=%s shape=%s' % (dim(embeddings), shape))
    return model


build_lstm = {
    1: build_lstm1,
    2: build_lstm2,
    3: build_lstm3,
    4: build_lstm4,
    5: build_lstm5,
    6: build_lstm6,
    7: build_lstm7,
    8: build_lstm8,
}


def compile_lstm(model, learn_rate):
    model.compile(optimizer=Adam(lr=learn_rate), loss='binary_crossentropy',
                  metrics=['accuracy'])
    xprint('compile_lstm: learn_rate=%g' % learn_rate)
    model.summary(print_fn=xprint)
    return model


def get_embeddings(vocab):
    return vocab.vectors.data


def predict(model_dir, texts, method, max_length, frozen):
    nlp = spacy.load('en_core_web_lg')
    print('----- pipe_names=%s' % nlp.pipe_names)
    nlp.pipeline = [
        ('tagger', nlp.tagger),
        ('parser', nlp.parser),
        ('sa', SentimentAnalyser.load(model_dir, nlp, method=method, max_length=max_length,
            frozen=frozen))
    ]
    print('+++++ pipe_names=%s' % nlp.pipe_names)

    y = np.zeros((len(texts), len(LABEL_COLS)), dtype=np.float32)
    for i, doc in enumerate(nlp.pipe(texts, batch_size=1000, n_threads=n_threads)):
        y[i, :] = doc.user_data['toxics']
    return y


def get_model_dir(model_name, fold):
    return os.path.join(MODEL_DIR, '%s.fold%d' % (model_name, fold))


class ClfSpacy:

    def __init__(self, n_hidden=64, max_length=100,  # Shape
        dropout=0.5, learn_rate=0.001,  # General NN config
        epochs=5, batch_size=100, frozen=True, lstm_type=1, predict_method=MEAN, force_fit=False):
        """
            n_hidden: Number of elements in the LSTM layer
            max_length: Max length of comment text
            max_features: Maximum vocabulary size
        """
        self.n_hidden = n_hidden
        self.max_length = max_length
        self.dropout = dropout
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.frozen = frozen
        self.lstm_type = lstm_type
        self.predict_method = predict_method

        self.description = ', '.join('%s=%s' % (k, v) for k, v in sorted(self.__dict__.items()))
        n_samples = str(N_SAMPLES) if N_SAMPLES >= 0 else 'ALL'
        self.model_name = 'lstm_spacy.%s_%03d_%03d_%.3f_%.3f.%s.%d' % (n_samples, n_hidden,
            max_length, dropout, learn_rate, frozen, lstm_type)
        self.force_fit = force_fit

        self.best_epochs = (-1, -1)
        assert lstm_type in build_lstm, self.description

    def __repr__(self):
        return 'ClfSpacy(%s)' % self.description

    def fit(self, train, test_size=0.1):
        model_dir = get_model_dir(self.model_name, 0)
        # RocAucEvaluation saves the trainable part of the model
        model_path = os.path.join(model_dir, 'model')
        config_path = os.path.join(model_dir, 'config.json')
        xprint('model_path=%s exists=%s' % (model_path, os.path.exists(model_path)))
        xprint('config_path=%s exists=%s' % (config_path, os.path.exists(config_path)))
        if os.path.exists(model_path) and os.path.exists(config_path) and not self.force_fit:
            xprint('model_path already exists. re-using')
            return
        os.makedirs(model_dir, exist_ok=True)

        xprint('ClfSpacy.fit: model_dir=%s' % model_dir)

        y_train = train[LABEL_COLS].values
        X_train = df_to_sentences(train)
        X_val, y_val = None, None
        if test_size > 0.0:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size)

        lstm_shape = {'n_hidden': self.n_hidden,
                      'max_length': self.max_length,
                      'n_class': len(LABEL_COLS)}
        lstm_settings = {'dropout': self.dropout,
                         'lr': self.learn_rate}
        lstm, self.best_epochs = do_train(X_train, y_train, X_val, y_val, lstm_shape, lstm_settings, {},
                        epochs=self.epochs, batch_size=self.batch_size, frozen=self.frozen,
                        lstm_type=self.lstm_type, model_path=model_path)

        with open(config_path, 'wt') as f:
            f.write(lstm.to_json())

        print('****: best_epochs=%s - %s' % (self.best_epochs, self.description))

    def predict(self, test):
        X_test = df_to_sentences(test)
        model_path = get_model_dir(self.model_name, 0)
        return predict(model_path, X_test, method=self.predict_method, max_length=self.max_length,
            frozen=self.frozen)
