import cytoolz
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Embedding, Bidirectional
from keras.layers import TimeDistributed
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from spacy.compat import pickle
import spacy
import os
import time
import multiprocessing
from framework import MODEL_DIR, LABEL_COLS, df_to_sentences, train_test_split
from utils import dim, xprint, RocAucEvaluation, Cycler
from spacy_glue import SpacySentenceCache


n_threads = max(multiprocessing.cpu_count() - 1, 1)
xprint('n_threads=%d' % n_threads)
sentence_cache = SpacySentenceCache()


if False:
    for lang in ('en', 'en_vectors_web_lg', 'en_core_web_lg'):
        nlp = spacy.load(lang)
        print('lang=%-16s pipe_names=%s' % (lang, nlp.pipe_names))
    assert False


class SentimentAnalyser(object):
    @classmethod
    def load(cls, path, nlp, max_length):
        xprint('SentimentAnalyser.load: path=%s max_length=%d' % (path, max_length))
        with open(os.path.join(path, 'config.json'), 'rt') as f:
            model = model_from_json(f.read())
        with open(os.path.join(path, 'model'), 'rb') as f:
            lstm_weights = pickle.load(f)
        embeddings = get_embeddings(nlp.vocab)
        model.set_weights([embeddings] + lstm_weights)
        return cls(model, max_length=max_length)

    def __init__(self, model, max_length):
        self._model = model
        self.max_length = max_length

    # def __call__(self, doc):
    #     X = get_features([doc], self.max_length)
    #     y = self._model.predict(X)
    #     self.set_sentiment(doc, y)

    def pipe(self, docs, batch_size=1000, n_threads=n_threads):
        for minibatch in cytoolz.partition_all(batch_size, docs):
            minibatch = list(minibatch)
            for doc in minibatch:
                Xs = get_features(doc.sents, self.max_length)
                ys = self._model.predict(Xs)
                doc.user_data['toxics'] = ys.mean(axis=0)
                yield doc

    # def set_sentiment(self, doc, y):
    #     print('set_sentiment: y=%s sentiment=%s' % (y, doc.sentiment))
    #     # doc.sentiment = float(y[0])
    #     doc.user_data['toxics'] = y
    #     assert False
    #     # Sentiment has a native slot for a single float.
    #     # For arbitrary data storage, there's:
    #     # doc.user_data['toxics'] = y


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
    model_path=None):
    """Train a Keras model on the sentences in `train_texts`
        All the sentences in a text have the text's label
    """

    print('do_train: train_texts=%s dev_texts=%s' % (dim(train_texts), dim(dev_texts)))

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
    model = compile_lstm(embeddings, lstm_shape, lstm_settings)

    callback_list = None
    if validation_data is not None:
        ra_val = RocAucEvaluation(validation_data=validation_data, interval=1, model_path=model_path)
        early = EarlyStopping(monitor='val_auc', mode='max', patience=3, verbose=1)
        callback_list = [ra_val, early]

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=validation_data, callbacks=callback_list, verbose=1)

    return model


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


def compile_lstm(embeddings, shape, settings):
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

    model.compile(optimizer=Adam(lr=settings['lr']), loss='binary_crossentropy',
                  metrics=['accuracy'])
    xprint('compile_lstm: embeddings=%s shape=%s' % (dim(embeddings), shape))
    xprint(model.summary())
    return model


def get_embeddings(vocab):
    return vocab.vectors.data


def predict(model_dir, texts, max_length):
    nlp = spacy.load('en_core_web_lg')
    print('----- pipe_names=%s' % nlp.pipe_names)
    nlp.pipeline = [
        ('tagger', nlp.tagger),
        ('parser', nlp.parser),
        ('sa', SentimentAnalyser.load(model_dir, nlp, max_length=max_length))
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
        epochs=5, batch_size=100, n_examples=-1):
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
        self.n_examples = n_examples

        self.description = ', '.join('%s=%s' % (k, v) for k, v in sorted(self.__dict__.items()))
        self.model_name = 'lstm_spacy_%03d_%03d_%.3f_%.3f' % (n_hidden, max_length, dropout, learn_rate)

    def __repr__(self):
        return 'ClfSpacy(%s)' % self.description

    def fit(self, train, test_size=0.1):
        model_dir = get_model_dir(self.model_name, 0)
        # RocAucEvaluation saves the trainable part of the mode;
        model_path = os.path.join(model_dir, 'model')
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
        lstm = do_train(X_train, y_train, X_val, y_val, lstm_shape, lstm_settings, {},
                        epochs=self.epochs, batch_size=self.batch_size, model_path=model_path)

        with open(os.path.join(model_dir, 'config.json'), 'wt') as f:
            f.write(lstm.to_json())

    def predict(self, test):
        X_test = df_to_sentences(test)
        return predict(get_model_dir(self.model_name, 0), X_test, max_length=self.max_length)
