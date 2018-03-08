import cytoolz
from collections import defaultdict
import numpy as np
from keras.models import Sequential
from keras.layers import (LSTM, Dense, Embedding, Bidirectional, Dropout, GlobalMaxPool1D,
    GlobalAveragePooling1D, BatchNormalization, TimeDistributed, Flatten)
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import spacy
import os
import time
import math
from framework import MODEL_DIR, LABEL_COLS, get_n_samples_str, df_to_sentences, train_test_split
from utils import (dim, xprint, RocAucEvaluation, SaveAllEpochs, Cycler, save_model, load_model,
    save_json, load_json)
from spacy_glue import SpacySentenceWordCache


PAD = 0
OOV = -1


sentence_cache = SpacySentenceWordCache()
if False:
    for lang in ('en', 'en_vectors_web_lg', 'en_core_web_lg'):
        nlp = spacy.load(lang)
        print('lang=%-16s pipe_names=%s' % (lang, nlp.pipe_names))
    assert False


MIN, MEAN, MAX, MEAN_MAX, MEDIAN, PC75, PC90, LINEAR, LINEAR2, LINEAR3, LINEAR4, LINEAR5, EXP = (
    'MIN', 'MEAN', 'MAX', 'MEAN_MAX', 'MEDIAN', 'PC75', 'PC90', 'LINEAR', 'LINEAR2', 'LINEAR3',
    'LINEAR4', 'LINEAR5', 'EXP')
PREDICT_METHODS = (MIN, MEAN, MAX, MEAN_MAX, MEDIAN, PC75, PC90, LINEAR, LINEAR2, LINEAR3, LINEAR4,
    LINEAR5, EXP)
PREDICT_METHODS_GOOD = [MEAN, LINEAR, LINEAR2, LINEAR3]


def linear_weights(ys, limit):
    """Returns: Array of linearly increasing weights [w_1, ..., w_n]
        n = len(ys)
        sum(weights) = 1.0
        w_1 = limit
        w_n = 1.0 - limit

        w_n / w_1 = (1 / limit - 1) increases as limit decreases
    """
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
    if method == MIN:
        return ys.min(axis=0)
    if method == MEAN:
        return ys.mean(axis=0)                 # w_n / w_1 = 1  4th BEST
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
        weights = linear_weights(ys, limit=0.1)  # w_n / w_1 = 9   3rd BEST
        return np.dot(weights, ys)
    elif method == LINEAR2:
        weights = linear_weights(ys, limit=0.2)  # w_n / w_1 = 4    BEST
        return np.dot(weights, ys)
    elif method == LINEAR3:
        weights = linear_weights(ys, limit=0.3)  # w_n / w_1 = 2.3  2nd BEST
        return np.dot(weights, ys)
    elif method == LINEAR4:
        weights = linear_weights(ys, limit=0.05)  # w_n / w_1 = 19
        return np.dot(weights, ys)
    elif method == LINEAR5:
        weights = linear_weights(ys, limit=0.01)  # w_n / w_1 = 99
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
    def load(cls, nlp, model_path, config_path, word_path, methods, max_length):
        xprint('SentimentAnalyser.load: model_path=%s config_path=%s word_path=%s methods=%s max_length=%d' % (
             model_path, config_path, word_path, methods, max_length))
        model = load_model(model_path, config_path)
        word_map = load_json(word_path)
        word_map = {int(k): v for k, v in word_map.items()}
        return cls(model, word_map, methods=methods, max_length=max_length)

    def __init__(self, model, word_map, methods, max_length):
        self._model = model
        self.word_map = word_map
        self.methods = methods
        self.max_length = max_length

    def __del__(self):
        del self._model

    def pipe(self, docs, batch_size=1000, n_threads=-1):
        interval = 10
        t0 = time.clock()
        i = 0
        k = 0
        for minibatch in cytoolz.partition_all(batch_size, docs):
            minibatch = list(minibatch)
            for doc in minibatch:
                Xs = get_features(self.word_map, doc.sents, self.max_length)
                ys = self._model.predict(Xs)
                if i >= interval:
                    xprint('SentimentAnalyser.pipe: %4d docs %5d sents %.1f sec' % (i, k, time.clock() - t0))
                    interval *= 2
                for method in self.methods:
                    y = reduce(ys, method=method)
                    assert len(y.shape) == 1 and len(y) == ys.shape[1], (ys.shape, y.shape)
                    doc.user_data[method] = y
                yield doc
                i += 1
                k += ys.shape[0]
        xprint('SentimentAnalyser.pipe: %4d docs %5d sents %.1f sec TOTAL' % (i, k, time.clock() - t0))


def word_count_add(word_count, wc):
    for w, c in wc.items():
        word_count[w] += c


def sentence_label_generator(texts_in, labels_in, batch_size, word_count):
    xprint('sentence_label_generator:', len(texts_in), len(labels_in), batch_size)
    sent_labels = []

    texts_cycler = Cycler(texts_in, batch_size)
    labels_cycler = Cycler(labels_in, batch_size)

    while True:
        texts = texts_cycler.batch()
        labels = labels_cycler.batch()
        text_sents, wc = sentence_cache.sent_id_pipe(texts)
        word_count_add(word_count, wc)
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
    word_count = defaultdict(int)
    xprint('make_sentences: Xs=%s ys=%s' % (dim(Xs), dim(ys)))

    gen = sentence_label_generator(texts_in, labels_in, batch_size, word_count)
    while n < n_sentences:
        sent_labels = next(gen)
        m = min(batch_size, n_sentences - n)
        # assert n + m <= n_sentences, (n, m, n_sentences)
        for i, (sent, y) in enumerate(sent_labels[:m]):
            # assert n + i < n_sentences, (n, i, n_sentences)
            for j, vector_id in enumerate(sent[:max_length]):
                # assert vector_id != 0
                Xs[n + i, j] = vector_id
                # assert Xs[n + i, j] in word_count, (n, i, j, Xs[n + i, j])
            word_count[PAD] += max(0, max_length - len(sent))
            ys[n + i, :] = y
        n += m
        if n % N == 0 or n + 1 == n_sentences:
            dt = max(time.clock() - t0, 1.0)
            print('^^^^%5s %7d (%5.1f%%) sents dt=%4.1f sec %3.1f sents/sec' %
                (name, n, 100.0 * n / n_sentences, dt, n / dt))

    # for i in range(Xs.shape[0]):
    #     for y in range(Xs.shape[1]):
    #         assert Xs[i, j] in word_count, (i, j, Xs[i, j])

    return Xs, ys, word_count


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


def do_fit(train_texts, train_labels, dev_texts, dev_labels, lstm_shape, lstm_settings,
    lstm_optimizer, batch_size=100, epochs=5,
    model_path=None, config_path=None, epoch_path=None, word_path=None,
    lstm_type=1, max_features=None):
    """Train a Keras model on the sentences in `train_texts`
        All the sentences in a text have the text's label
        do_fit1: Fit with frozen word embeddings
        do_fit2: Fit with unfrozen word embeddings (after fitting with frozen embeddings) at a lower
                learning rate
    """

    xprint('do_fit: train_texts=%s dev_texts=%s' % (dim(train_texts), dim(dev_texts)))
    best_epoch = -1

    n_train_sents = count_sentences(train_texts, batch_size, 'train')
    X_train, y_train, word_count = make_sentences(lstm_shape['max_length'], batch_size,
        train_texts, train_labels, 'train', n_train_sents)
    validation_data = None
    if dev_texts is not None:
        n_dev_sents = count_sentences(dev_texts, batch_size, 'dev')
        X_val, y_val, wc_val = make_sentences(lstm_shape['max_length'], batch_size,
            dev_texts, dev_labels, 'dev', n_dev_sents)
        validation_data = (X_val, y_val)
    sentence_cache.flush()

    print("Loading spaCy")
    nlp = sentence_cache._load_nlp()
    word_list, word_map = make_word_map(word_count, max_features)
    save_json(word_path, word_map)
    embeddings = get_embeddings(nlp.vocab, word_list)

    X_train = reindex(word_map, X_train)
    if validation_data is not None:
        X_val = reindex(word_map, X_val)
        validation_data = (X_val, y_val)

    print('reindexed')

    model = build_lstm[lstm_type](embeddings, lstm_shape, lstm_settings)
    compile_lstm(model, lstm_settings['lr'])

    print('built and compiled models')

    # for i in range(X_train.shape[0]):
    #     for j in range(X_train.shape[1]):
    #         assert 0 <= X_train[i, j] < embeddings.shape[0], (i, j, X_train[i, j], dim(embeddings))
    # for i in range(X_val.shape[0]):
    #     for j in range(X_val.shape[1]):
    #         assert 0 <= X_val[i, j] < embeddings.shape[0], (i, j, X_val[i, j], dim(embeddings))

    print('^^^embeddings=%s' % dim(embeddings))
    print('^^^X_train=%d..%d' % (X_train.min(), X_train.max()))
    print('^^^X_val=%d..%d' % (X_val.min(), X_val.max()))

    callback_list = None

    if validation_data is not None:
        ra_val = RocAucEvaluation(validation_data=validation_data, interval=1,
            model_path=model_path, config_path=config_path)
        early = EarlyStopping(monitor='val_auc', mode='max', patience=2, verbose=1)
        callback_list = [ra_val, early]
    else:
        sae = SaveAllEpochs(model_path, config_path, epoch_path, True)
        if sae.last_epoch1() > 0:
            xprint('Reloading partially built model 1')
            model = load_model(model_path, config_path)
            compile_lstm(model, lstm_settings['lr'])
            epochs -= sae.last_epoch1()
        callback_list = [sae]

    if epochs > 0:
        print('!^^^embeddings=%s' % dim(embeddings))
        print('!^^^X_train=%d..%d' % (X_train.min(), X_train.max()))
        print('!^^^X_val=%d..%d' % (X_val.min(), X_val.max()))

        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list,
                  validation_data=validation_data, verbose=1)
        if validation_data is not None:
            best_epoch = ra_val.best_epoch
            ra_val.best_epoch = -1
        else:
            save_model(model, model_path, config_path)

    del nlp
    return model, best_epoch,


def get_features(word_map, docs, max_length):
    """This works because PAD == 0
    """
    docs = list(docs)
    oov_idx = word_map[OOV]
    Xs = np.ones((len(docs), max_length), dtype='int32') * word_map[PAD]
    for i, doc in enumerate(docs):
        for j, token in enumerate(doc):
            if j >= max_length:
                break
            vector_id = token.vocab.vectors.find(key=token.orth)
            Xs[i, j] = word_map.get(vector_id, oov_idx)
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
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False), name='td2'))
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
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False, name='td3')))
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
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False, name='td4')))
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
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False), name='td6'))
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
    """RELU dense layer
    """
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
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False), name='td7'))
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
    """Flatten rather than pool"""
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
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False, name='td8')))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout']), name='bidi'))
    model.add(Flatten(name='flaaten'))
    model.add(BatchNormalization())
    model.add(Dropout(settings['dropout'] / 2.0))
    model.add(Dense(shape['n_class'], activation='sigmoid'))
    xprint('build_lstm8: embeddings=%s shape=%s' % (dim(embeddings), shape))
    return model


def build_lstm9(embeddings, shape, settings):
    """2 layer LSTM
    """
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
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False), name='td9a'))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout']), name='bidi9a'))
    # model.add(GlobalMaxPool1D())
    # model.add(BatchNormalization())
    # model.add(Dropout(settings['dropout'] / 2.0))

    # model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False), name='td9b'))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout']), name='bidi9b'))
    model.add(GlobalMaxPool1D(name='mp9'))
    model.add(BatchNormalization(name='bn9'))
    model.add(Dropout(settings['dropout'] / 2.0, name='drop9b'))

    model.add(Dense(shape['n_class'], activation='sigmoid', name='den9b'))
    xprint('build_lstm9: embeddings=%s shape=%s' % (dim(embeddings), shape))
    return model


def build_lstm10(embeddings, shape, settings):
    """3 layer LSTM
    """
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
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False), name='td9a'))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout']), name='bidi9a'))
    # model.add(GlobalMaxPool1D())
    # model.add(BatchNormalization())
    # model.add(Dropout(settings['dropout'] / 2.0))

    # model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False), name='td9b'))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout']), name='bidi9b'))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout']), name='bidi9c'))
    model.add(GlobalMaxPool1D(name='mp9'))
    model.add(BatchNormalization(name='bn9'))
    model.add(Dropout(settings['dropout'] / 2.0, name='drop9b'))

    model.add(Dense(shape['n_class'], activation='sigmoid', name='den9b'))
    xprint('build_lstm9: embeddings=%s shape=%s' % (dim(embeddings), shape))
    return model


def build_lstm11(embeddings, shape, settings):
    """3 layer LSTM with fewer weights in higher layers
    """
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
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False), name='td9a'))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout']), name='bidi9a'))
    # model.add(GlobalMaxPool1D())
    # model.add(BatchNormalization())
    # model.add(Dropout(settings['dropout'] / 2.0))

    # model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False), name='td9b'))
    model.add(Bidirectional(LSTM(shape['n_hidden'] // 2, return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout']), name='bidi9b'))
    model.add(Bidirectional(LSTM(shape['n_hidden'] // 2, return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout']), name='bidi9c'))
    model.add(GlobalMaxPool1D(name='mp9'))
    model.add(BatchNormalization(name='bn9'))
    model.add(Dropout(settings['dropout'] / 2.0, name='drop9b'))

    model.add(Dense(shape['n_class'], activation='sigmoid', name='den9b'))
    xprint('build_lstm9: embeddings=%s shape=%s' % (dim(embeddings), shape))
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
    9: build_lstm9,
    10: build_lstm10,
    11: build_lstm11,
}


def compile_lstm(model, learn_rate):
    model.compile(optimizer=Adam(lr=learn_rate), loss='binary_crossentropy', metrics=['accuracy'])
    xprint('compile_lstm: learn_rate=%g' % learn_rate)
    model.summary(print_fn=xprint)
    return model


def make_word_map(word_count, max_features, verbose=True):
    xprint('make_word_map: word_count=%d max_features=%d' % (len(word_count), max_features))
    ordinary = [w for w in word_count if w > 0]

    # Keep the most commmon `max_features` words in the embedding
    # words = [-1 (OOV), 0 (PAD), word ids]
    # special = words[:2]
    # ordinary = words[2:]
    ordinary.sort(key=lambda w: (-word_count[w], w))
    ordinary_keep = ordinary[:max_features - 2]
    word_list = [OOV, PAD] + ordinary_keep
    word_map = {w: i for i, w in enumerate(word_list)}

    if verbose:
        wc_all = {w: word_count[w] for w in ordinary}
        wc_keep = {w: word_count[w] for w in ordinary_keep}
        n_a = len(wc_all)
        m_a = sum(wc_all.values())
        r_a = m_a / n_a
        n_k = len(wc_keep)
        m_k = sum(wc_keep.values())
        r_k = m_k / n_k
        xprint('  all : unique=%d total=%d ave repeats=%.1f' % (n_a, m_a, r_a))
        xprint('  keep: unique=%d total=%d ave repeats=%.1f' % (n_k, m_k, r_k))
        xprint('  frac: unique=%.3f total=%.3f repeats=%.3f' % (n_k / n_a, m_k / m_a, r_k / r_a))

    return word_list, word_map


def get_embeddings(vocab, word_list):
    embed_size = vocab.vectors.data.shape[1]
    emb_mean, emb_std = vocab.vectors.data.mean(), vocab.vectors.data.std()
    xprint('emb_mean=%.3f emb_std=%.3f' % (emb_mean, emb_std))

    embeddings = np.empty((len(word_list), embed_size), dtype=np.float32)
    embeddings[0, :] = np.random.normal(emb_mean, emb_std, embed_size)   # OOV
    embeddings[1, :] = np.zeros(embed_size, dtype=np.float32)  # PAD
    embeddings[2:, :] = vocab.vectors.data[word_list[2:], :]

    xprint('get_embeddings: vocab=%s word_list=%d -> embeddings=%s' % (
        dim(vocab.vectors.data), len(word_list), dim(embeddings)))

    return embeddings


def reindex(word_map, X):
    oov_idx = word_map[OOV]
    assert oov_idx == 0, oov_idx
    values = list(word_map.values())
    assert min(values) == 0, min(values)
    assert max(values) == len(word_map) - 1, (max(values), len(word_map))

    X2 = X.copy()
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X2[i, j] = word_map.get(X[i, j], oov_idx)

    # for i in range(X.shape[0]):
    #     for j in range(X.shape[1]):
    #         assert 0 <= X2[i, j] < len(word_map), ([i, j], X2[i, j], len(word_map))
    return X2


def predict_reductions(model_path, config_path, word_path, texts, methods, max_length):
    xprint('predict_reductions(model_path=%s, config_path=%s, word_path=%s, texts=%s, methods=%s, '
          'max_length=%d)' %
          (model_path, config_path, word_path, dim(texts), methods, max_length))

    nlp = spacy.load('en_core_web_lg')
    print('----- pipe_names=%s' % nlp.pipe_names)
    nlp.pipeline = [
        ('tagger', nlp.tagger),
        ('parser', nlp.parser),
        ('sa', SentimentAnalyser.load(nlp, model_path, config_path, word_path, methods=methods,
                                      max_length=max_length))
    ]
    print('+++++ pipe_names=%s' % nlp.pipe_names)

    reductions = {method: np.zeros((len(texts), len(LABEL_COLS)), dtype=np.float32)
                  for method in methods}
    for i, doc in enumerate(nlp.pipe(texts, batch_size=1000)):
        for method in methods:
            reductions[method][i, :] = doc.user_data[method]

    del nlp
    return reductions


def get_model_dir(model_name, fold):
    return os.path.join(MODEL_DIR, '%s.fold%d' % (model_name, fold))


class ClfSpacy:

    def __init__(self, n_hidden=64, max_length=100, max_features=20000,  # Shape
        dropout=0.5, learn_rate=0.001,  # General NN config
        epochs=5, batch_size=100, lstm_type=1, predict_method=MEAN, force_fit=False):
        """
            n_hidden: Number of elements in the LSTM layer
            max_length: Max length of comment text
            max_features: Maximum vocabulary size

            frozen => freeze embeddings
            2 stages of training: frozen, unfrozen


        """
        self.n_hidden = n_hidden
        self.max_length = max_length
        self.dropout = dropout
        self.learn_rate = learn_rate
        self.max_features = max_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.lstm_type = lstm_type
        self.predict_method = predict_method

        self.description = ', '.join('%s=%s' % (k, v) for k, v in sorted(self.__dict__.items()))
        self.model_name = 'lstm_spacy_e.%s.%03d_%03d_%05d_%.3f_%.3f.%s.epochs%d' % (
            get_n_samples_str(),
            n_hidden, max_length, max_features,
            dropout, learn_rate, lstm_type, epochs)

        self.force_fit = force_fit
        self._shown_paths = False

        self.best_epoch = -1
        assert lstm_type in build_lstm, self.description

    def __repr__(self):
        return 'ClfSpacy(%s)' % self.description

    def _get_paths(self, create_dir):
        model_dir = get_model_dir(self.model_name, 0)
        if create_dir:
            os.makedirs(model_dir, exist_ok=True)
        # RocAucEvaluation saves the trainable part of the model
        model_path = os.path.join(model_dir, 'model')
        config_path = os.path.join(model_dir, 'config.json')
        word_path = os.path.join(model_dir, 'words.json')
        epoch_path = os.path.join(model_dir, 'epochs.json')
        if not self._shown_paths:
            xprint('model_path=%s exists=%s' % (model_path, os.path.exists(model_path)))
            xprint('config_path=%s exists=%s' % (config_path, os.path.exists(config_path)))
            xprint('word_path=%s exists=%s' % (word_path, os.path.exists(word_path)))
            xprint('epoch_path=%s exists=%s' % (epoch_path, os.path.exists(epoch_path)))
            self._shown_paths = True
        return model_path, config_path, word_path, epoch_path

    def fit(self, train, test_size=0.1):
        xprint('ClfSpacy.fit', '-' * 80)
        model_path, config_path, word_path, epoch_path = self._get_paths(True)
        if not self.force_fit:
            if (os.path.exists(model_path) and os.path.exists(config_path) and
                os.path.exists(word_path) and
                SaveAllEpochs.epoch_dict(epoch_path)['epoch1'] == self.epochs):
                xprint('model_path already exists. re-using')
                return

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
        lstm, self.best_epoch = do_fit(X_train, y_train, X_val, y_val, lstm_shape,
            lstm_settings, {},
            batch_size=self.batch_size, lstm_type=self.lstm_type, epochs=self.epochs,
            model_path=model_path, config_path=config_path, word_path=word_path, epoch_path=epoch_path,
            max_features=self.max_features)

        assert isinstance(self.best_epoch, int), self.best_epoch
        xprint('****: best_epoch=%s - %s Add 1 to this' % (self.best_epoch, self.description))
        del lstm

    def predict_reductions(self, test, predict_methods):
        print('ClfSpacy.predict_reductions', '-' * 80)
        X_test = df_to_sentences(test)
        model_path, config_path, word_path, _ = self._get_paths(False)

        assert os.path.exists(model_path), model_path
        assert os.path.exists(config_path), config_path

        return predict_reductions(model_path, config_path, word_path, X_test,
            predict_methods, self.max_length,)

    def predict(self, test):
        reductions = self.predict_reductions(test, predict_methods=[self.predict_method])
        print('ClfSpacy.predict', '-' * 80)
        return reductions[self.predict_method]
