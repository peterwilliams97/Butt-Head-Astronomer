import random
import cytoolz
import numpy
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Embedding, Bidirectional
from keras.layers import TimeDistributed
from keras.optimizers import Adam
# import keras.backend as K
from spacy.compat import pickle
import spacy

import os
import numpy as np
import multiprocessing
from framework import MODEL_DIR, LABEL_COLS, df_to_sentences, train_test_split
from utils import dim, xprint


n_threads = max(multiprocessing.cpu_count() - 1, 1)
xprint('n_threads=%d' % n_threads)

if False:
    for lang in ('en', 'en_vectors_web_lg', 'en_core_web_lg'):
        nlp = spacy.load(lang)
        print('lang=%-16s pipe_names=%s' % (lang, nlp.pipe_names))
    assert False


class SentimentAnalyser(object):
    @classmethod
    def load(cls, path, nlp, max_length=100):
        xprint('SentimentAnalyser.load: path=%s max_length=%d' % (path, max_length))
        with open(os.path.join(path, 'config.json'), 'rt') as f:
            model = model_from_json(f.read())
        with open(os.path.join(path, 'model'), 'rb') as f:
            lstm_weights = pickle.load(f)
        embeddings = get_embeddings(nlp.vocab)

        xprint('d' * 80)
        xprint('embeddings=%s' % dim(embeddings))
        xprint('lstm_weights=%s' % dim(lstm_weights))
        xprint('e' * 80)

        xprint('SentimentAnalyser.load')
        xprint(model.summary())
        model.set_weights([embeddings] + lstm_weights)
        return cls(model, max_length=max_length)

    def __init__(self, model, max_length=100):
        self._model = model
        self.max_length = max_length

    def __call__(self, doc):
        X = get_features([doc], self.max_length)
        y = self._model.predict(X)
        self.set_sentiment(doc, y)

    def pipe(self, docs, batch_size=1000, n_threads=2):
        for minibatch in cytoolz.partition_all(batch_size, docs):
            minibatch = list(minibatch)
            sentences = []
            for doc in minibatch:
                sentences.extend(doc.sents)
                doc.user_data['toxics'] = np.zeros(len(LABEL_COLS), dtype=np.float32)
            Xs = get_features(sentences, self.max_length)
            ys = self._model.predict(Xs)
            # print('pipe: sentences=%s sentences[0]=%s Xs=%s ys=%s' % (
            #     dim(sentences), dim(sentences[0]), dim(Xs), dim(ys)))
            for sent, label in zip(sentences, ys):
                # print('pipe: sent=%s label=%s' % (sent, label))
                sent.doc.user_data['toxics'] += label - 0.5
            for doc in minibatch:
                yield doc

    def set_sentiment(self, doc, y):
        print('set_sentiment: y=%s sentiment=%s' % (y, doc.sentiment))
        # doc.sentiment = float(y[0])
        doc.user_data['toxics'] = y
        assert False
        # Sentiment has a native slot for a single float.
        # For arbitrary data storage, there's:
        # doc.user_data['toxics'] = y


def get_labelled_sentences(docs, doc_labels):
    labels = []
    sentences = []
    for doc, y in zip(docs, doc_labels):
        for sent in doc.sents:
            sentences.append(sent)
            labels.append(y)
    return sentences, numpy.asarray(labels, dtype='int32')


def do_train(train_texts, train_labels, dev_texts, dev_labels,
    lstm_shape, lstm_settings, lstm_optimizer, batch_size=100, epochs=5, by_sentence=True):
    """Train a Keras model on the sentences in `train_texts`
        All the sentences in a text have the text's label
    """

    print("Loading spaCy")
    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    print("Parsing texts...")
    train_docs = list(nlp.pipe(train_texts))
    dev_docs = list(nlp.pipe(dev_texts))
    if by_sentence:
        train_docs, train_labels = get_labelled_sentences(train_docs, train_labels)
        dev_docs, dev_labels = get_labelled_sentences(dev_docs, dev_labels)

    train_X = get_features(train_docs, lstm_shape['max_length'])
    dev_X = get_features(dev_docs, lstm_shape['max_length'])
    print('do_train: train_X=%s dev_X=%s' % (dim(train_X), dim(dev_X)))

    embeddings = get_embeddings(nlp.vocab)
    model = compile_lstm(embeddings, lstm_shape, lstm_settings)
    model.fit(train_X, train_labels, validation_data=(dev_X, dev_labels),
              epochs=epochs, batch_size=batch_size)
    return model


def get_features(docs, max_length):
    docs = list(docs)
    Xs = numpy.zeros((len(docs), max_length), dtype='int32')
    for i, doc in enumerate(docs):
        for j, token in enumerate(doc):
            vector_id = token.vocab.vectors.find(key=token.orth)
            if vector_id >= 0:
                Xs[i, j] = vector_id
            # else:
            #     Xs[i, j] = 0
            if j > max_length:
                break
    print('get_features: Xs=%s' % dim(Xs))
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
    model.add(TimeDistributed(Dense(shape['nr_hidden'], use_bias=False)))
    model.add(Bidirectional(LSTM(shape['nr_hidden'],
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout'])))
    model.add(Dense(shape['nr_class'], activation='sigmoid'))
    model.compile(optimizer=Adam(lr=settings['lr']), loss='binary_crossentropy',
                  metrics=['accuracy'])
    xprint('compile_lstm: embeddings=%s shape=%s' % (dim(embeddings), shape))
    xprint(model.summary())
    return model


def get_embeddings(vocab):
    embeddings = vocab.vectors.data
    assert embeddings.any()
    return vocab.vectors.data


def predict(model_dir, texts, max_length=100):
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


def evaluate(model_dir, texts, labels, max_length=100):
    y = predict(model_dir, texts, max_length)
    n = len(y)
    return sum(bool(y[i] >= 0.5) == bool(labels[i]) for i in range(n)) / n


def read_data(data_dir, limit=0):
    examples = []
    for subdir, label in (('pos', 1), ('neg', 0)):
        for filename in (data_dir / subdir).iterdir():
            with filename.open() as file_:
                text = file_.read()
            examples.append((text, label))
    random.shuffle(examples)
    if limit >= 1:
        examples = examples[:limit]
    return zip(*examples)  # Unzips into two lists


def get_model_dir(model_name, fold):
    return os.path.join(MODEL_DIR, '%s.fold%d' % (model_name, fold))


class ClfSpacy:

    def __init__(self, n_hidden=64, max_length=100,  # Shape
        dropout=0.5, learn_rate=0.001,  # General NN config
        epochs=5, batch_size=100, n_examples=-1):
        """
            embed_size: Size of embedding vectors
            maxlen: Max length of comment text
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

    def fit(self, train):
        y_train0 = train[LABEL_COLS].values
        X_train0 = df_to_sentences(train)
        X_train, X_val, y_train, y_val = train_test_split(X_train0, y_train0, test_size=0.1)

        lstm_shape = {'nr_hidden': self.n_hidden,
                      'max_length': self.max_length,
                      'nr_class': len(LABEL_COLS)}
        lstm_settings = {'dropout': self.dropout,
                         'lr': self.learn_rate}
        lstm = do_train(X_train, y_train, X_val, y_val, lstm_shape, lstm_settings, {},
                        epochs=self.epochs, batch_size=self.batch_size)
        weights = lstm.get_weights()

        if False:
            nlp = spacy.load('en_core_web_lg')
            nlp.add_pipe(nlp.create_pipe('sentencizer'))
            embeddings = get_embeddings(nlp.vocab)
            lstm_weights = weights[1:]
            xprint('D' * 80)
            xprint('embeddings=%s' % dim(embeddings))
            xprint('weights=%s' % dim(weights))
            xprint('lstm_weights=%s' % dim(lstm_weights))
            xprint('E' * 80)

            model = lstm
            xprint(model.summary())
            print('!' * 80)
            model.set_weights([embeddings] + lstm_weights)
            xprint(model.summary())
            print('*' * 80)

        model_dir = get_model_dir(self.model_name, 0)
        os.makedirs(model_dir, exist_ok=True)
        xprint('ClfSpacy.fit: model_dir=%s' % model_dir)
        with open(os.path.join(model_dir, 'model'), 'wb') as f:
            pickle.dump(weights[1:], f)
        with open(os.path.join(model_dir, 'config.json'), 'wt') as f:
            f.write(lstm.to_json())

    def predict(self, test):
        X_test = df_to_sentences(test)
        return predict(get_model_dir(self.model_name, 0), X_test, max_length=100)


# @plac.annotations(
#     train_dir=("Location of training file or directory"),
#     dev_dir=("Location of development file or directory"),
#     model_dir=("Location of output model directory",),
#     is_runtime=("Demonstrate run-time usage", "flag", "r", bool),
#     nr_hidden=("Number of hidden units", "option", "H", int),
#     max_length=("Maximum sentence length", "option", "L", int),
#     dropout=("Dropout", "option", "d", float),
#     learn_rate=("Learn rate", "option", "e", float),
#     epochs=("Number of training epochs", "option", "i", int),
#     batch_size=("Size of minibatches for training LSTM", "option", "b", int),
#     nr_examples=("Limit to N examples", "option", "n", int)
# )
# def main(model_dir=None, train_dir=None, dev_dir=None,
#          is_runtime=False,
#          nr_hidden=64, max_length=100,  # Shape
#          dropout=0.5, learn_rate=0.001,  # General NN config
#          epochs=5, batch_size=100, nr_examples=-1):  # Training params
#     if model_dir is not None:
#         model_dir = pathlib.Path(model_dir)
#     if train_dir is None or dev_dir is None:
#         imdb_data = thinc.extra.datasets.imdb()
#     if is_runtime:
#         if dev_dir is None:
#             dev_texts, dev_labels = zip(*imdb_data[1])
#         else:
#             dev_texts, dev_labels = read_data(dev_dir)
#         acc = evaluate(model_dir, dev_texts, dev_labels, max_length=max_length)
#         print(acc)
#     else:
#         if train_dir is None:
#             train_texts, train_labels = zip(*imdb_data[0])
#         else:
#             print("Read data")
#             train_texts, train_labels = read_data(train_dir, limit=nr_examples)
#         if dev_dir is None:
#             dev_texts, dev_labels = zip(*imdb_data[1])
#         else:
#             dev_texts, dev_labels = read_data(dev_dir, imdb_data, limit=nr_examples)
#         train_labels = numpy.asarray(train_labels, dtype='int32')
#         dev_labels = numpy.asarray(dev_labels, dtype='int32')
#         lstm = train(train_texts, train_labels, dev_texts, dev_labels,
#                      {'nr_hidden': nr_hidden, 'max_length': max_length, 'nr_class': 1},
#                      {'dropout': dropout, 'lr': learn_rate},
#                      {},
#                      epochs=epochs, batch_size=batch_size)
#         weights = lstm.get_weights()
#         if model_dir is not None:
#             with (model_dir / 'model').open('wb') as file_:
#                 pickle.dump(weights[1:], file_)
#             with (model_dir / 'config.json').open('wb') as file_:
#                 file_.write(lstm.to_json())
