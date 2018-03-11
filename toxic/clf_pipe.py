import numpy as np
from keras.callbacks import EarlyStopping
import os
import time
import random
from framework import MODEL_DIR, LABEL_COLS, get_n_samples_str, df_to_sentences, train_test_split
from utils import (dim, xprint, RocAucEvaluation, SaveAllEpochs, save_model, load_model,
    save_json, load_json)
from token_spacy import SpacySentenceTokenizer, islowercase
from lstm import build_lstm, compile_lstm
from reductions import reduce, MEAN
from embeddings import get_embeddings, PAD, OOV


tokenizer = SpacySentenceTokenizer()


class ClfPipe:

    def __init__(self, n_hidden=64, max_length=100, max_features=20000,  # Shape
        embed_name=None, embed_size=None,
        dropout=0.5, learn_rate=0.001, learn_rate_unfrozen=0.0, frozen=False,  # General NN config
        epochs=5, batch_size=100, lstm_type=1, predict_method=MEAN, force_fit=False):
        """
            n_hidden: Number of elements in the LSTM layer
            max_length: Max length of comment text
            max_features: Maximum vocabulary size

            frozen => freeze embeddings
            2 stages of training: frozen, unfrozen

        """
        assert embed_name is not None
        self.n_hidden = n_hidden
        self.max_length = max_length
        self.dropout = dropout
        self.learn_rate = learn_rate
        self.learn_rate_unfrozen = learn_rate_unfrozen
        self.frozen = frozen
        self.max_features = max_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.lstm_type = lstm_type
        self.predict_method = predict_method

        self.embed_name, self.embed_size = embed_name, embed_size
        self.lowercase = self.embed_name != '840B'

        D = self.__dict__
        self.description = ', '.join('%s=%s' % (k, v) for k, v in sorted(D.items()))
        self.model_name = 'pipe.%s' % '-'.join(str(D[k]) for k in sorted(D))

        self.force_fit = force_fit
        self._shown_paths = False

        self.best_epoch = -1
        assert lstm_type in build_lstm, self.description

    def __repr__(self):
        return 'ClfPipe(%s)' % self.description

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
        xprint('ClfPipe.fit', '-' * 80)

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
                         'lr': self.learn_rate,
                         'lr_unfrozen': self.learn_rate_unfrozen}
        lstm, self.best_epochs = do_fit(X_train, y_train, X_val, y_val, lstm_shape,
            lstm_settings, {}, frozen=self.frozen,
            batch_size=self.batch_size, lstm_type=self.lstm_type, epochs=self.epochs,
            model_path=model_path, config_path=config_path, word_path=word_path, epoch_path=epoch_path,
            max_features=self.max_features,
            embed_name=self.embed_name, embed_size=self.embed_size, lowercase=self.lowercase)

        assert isinstance(self.best_epochs, dict), self.best_epochs
        xprint('****: best_epochs=%s - %s Add 1 to this' % (self.best_epochs, self.description))
        del lstm

    def predict_reductions(self, test, predict_methods):
        print('ClfPipe.predict_reductions', '-' * 80)
        X_test = df_to_sentences(test)
        model_path, config_path, word_path, _ = self._get_paths(False)

        assert os.path.exists(model_path), model_path
        assert os.path.exists(config_path), config_path

        return predict_reductions(model_path, config_path, word_path, X_test,
            predict_methods, self.max_length, self.lowercase)

    def predict(self, test):
        reductions = self.predict_reductions(test, predict_methods=[self.predict_method])
        print('ClfPipe.predict', '-' * 80)
        return reductions[self.predict_method]


def do_fit(train_texts, train_labels, dev_texts, dev_labels, lstm_shape, lstm_settings,
    lstm_optimizer, batch_size=100, epochs=5, frozen=True,
    model_path=None, config_path=None, epoch_path=None, word_path=None,
    lstm_type=1, max_features=None, embed_name=None, embed_size=None, lowercase=False):
    """Train a Keras model on the sentences in `train_texts`
        All the sentences in a text have the text's label
    """
    max_length = lstm_shape['max_length']

    xprint('do_fit: train_texts=%s dev_texts=%s' % (dim(train_texts), dim(dev_texts)))
    best_epochs = {}

    X_sents, word_list, word_index = extract_sentences(max_features, train_texts, lowercase, 'train')
    validation_data = None
    if dev_texts is not None:
        y_sents, _, _ = extract_sentences(max_features, dev_texts, lowercase, 'dev')
    tokenizer.flush()

    embeddings, reduced_index = get_embeddings(embed_name, embed_size, max_features, word_list, word_index)
    assert 0 <= min(reduced_index.values()) and max(reduced_index.values()) < embeddings.shape[0]
    save_json(word_path, reduced_index)
    del word_index

    X_train, y_train = apply_word_index(max_length, X_sents, reduced_index, train_texts, train_labels, 'train')
    if dev_texts is not None:
        X_val, y_val = apply_word_index(max_length, y_sents, reduced_index, train_texts, train_labels, 'dev')
        validation_data = (X_val, y_val)

    print('^^^embeddings=%s' % dim(embeddings))
    print('^^^X_train=%d..%d' % (X_train.min(), X_train.max()))
    print('^^^X_val=%d..%d' % (X_val.min(), X_val.max()))
    assert 0 <= X_train.min() and X_train.max() < embeddings.shape[0]
    assert 0 <= X_val.min() and X_val.max() < embeddings.shape[0]

    model = build_lstm[lstm_type](embeddings, lstm_shape, lstm_settings)

    xprint('built and compiled models')

    param_list = [(lstm_settings['lr'], True)]
    best_epochs = {}
    if not frozen:
        param_list.append((lstm_settings['lr_unfrozen'], False))

    for run, (learning_rate, frozen) in enumerate(param_list):
        xprint('do_fit: run=%d learning_rate=%g frozen=%s' % (run, learning_rate, frozen))
        if run > 0:
            xprint('Reloading partially stopped model')
            model = load_model(model_path, config_path)

        compile_lstm(model, learning_rate, frozen)
        callback_list = None

        if validation_data is not None:
            ra_val = RocAucEvaluation(validation_data=validation_data, interval=1,
                model_path=model_path, config_path=config_path, do_prime=run > 0)
            early = EarlyStopping(monitor='val_auc', mode='max', patience=2, verbose=1)
            callback_list = [ra_val, early]
        else:
            sae = SaveAllEpochs(model_path, config_path, epoch_path, True)
            if sae.last_epoch1() > 0:
                xprint('Reloading partially built model 1')
                model = load_model(model_path, config_path)
                compile_lstm(model, learning_rate, frozen)
                epochs -= sae.last_epoch1()
            callback_list = [sae]

        if epochs > 0:
            print('!^^^embeddings=%s' % dim(embeddings))
            print('!^^^X_train=%d..%d' % (X_train.min(), X_train.max()))
            print('!^^^X_val=%d..%d' % (X_val.min(), X_val.max()))

            model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list,
                      validation_data=validation_data, verbose=1)
            if validation_data is not None:
                best_epochs[run] = ra_val.best_epoch
                ra_val.best_epoch = -1
            else:
                save_model(model, model_path, config_path)

    return model, best_epochs


def extract_sentences(max_features, texts_in, lowercase, name):

    sents_list, word_count = tokenizer.sentence_tokens(texts_in, lowercase)
    word_list, word_index = make_word_index(word_count, max_features)

    for i, w in enumerate(word_list[2:]):
        assert islowercase(w), (i, w, lowercase)

    return sents_list, word_list, word_index


def apply_word_index(max_length, sents_list, word_index, texts_in, labels_in, name):
    n_sentences = sum(len(sents) for sents in sents_list)

    # PAD = 0 so Xs is all PADs to start with
    Xs = np.zeros((n_sentences, max_length), dtype='int32')
    ys = np.zeros((n_sentences, len(LABEL_COLS)), dtype='int32')
    xprint('extract_sentences: Xs=%s ys=%s' % (dim(Xs), dim(ys)))

    order = list(range(n_sentences))
    random.shuffle(order)
    oov_idx = word_index[OOV]

    t0 = time.clock()
    N = max(10000, n_sentences // 5)
    # assert N > 1000, (n_sentences, N)
    i = 0
    for text, sents, y in zip(texts_in, sents_list, labels_in):
        for sent in sents:
            for j, word in enumerate(sent[:max_length]):
                Xs[order[i], j] = word_index.get(word, oov_idx)
            ys[order[i]] = y
            if i % N == 0 or (i + 1) == n_sentences:
                dt = max(time.clock() - t0, 1.0)
                if dt > 2.0:
                    print('~~~%5s %7d (%5.1f%%) sents dt=%4.1f sec %3.1f sents/sec' %
                         (name, i, 100.0 * i / n_sentences, dt, i / dt))
            i += 1

    return Xs, ys


def make_word_index(word_count, max_features, verbose=True):
    xprint('make_word_index: word_count=%d max_features=%d' % (len(word_count), max_features))
    ordinary = list(word_count)

    # Keep the most commmon `max_features` words in the embedding
    # words = [-1 (OOV), 0 (PAD), word ids]
    # special = words[:2]
    # ordinary = words[2:]
    ordinary.sort(key=lambda w: (-word_count[w], w))
    word_list = [OOV, PAD] + ordinary
    word_index = {w: i for i, w in enumerate(word_list)}

    if verbose:
        ordinary_keep = ordinary[:max_features - 2]
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

    return word_list, word_index


def predict_reductions(model_path, config_path, word_path, texts, methods, max_length, lowercase):
    text_score = predict_scores(model_path, config_path, word_path, texts, max_length, lowercase)

    reductions = {method: np.zeros((len(texts), len(LABEL_COLS)), dtype=np.float32)
                  for method in methods}
    for i, text in enumerate(texts):
        ys = text_score[text]
        for method in methods:
            reductions[method][i, :] = reduce(ys, method)

    return reductions


def predict_scores(model_path, config_path, word_path, texts_in, max_length, lowercase):
    xprint('predict_reductions(model_path=%s, config_path=%s, word_path=%s, texts=%s, max_length=%d)'
           % (model_path, config_path, word_path, dim(texts_in), max_length))

    model = load_model(model_path, config_path)
    word_index = load_json(word_path)
    oov_idx = word_index[OOV]

    sents_list, word_count = tokenizer.sentence_tokens(texts_in, lowercase)
    text_score = {}
    for text, sents in zip(texts_in, sents_list):
        Xs = np.ones((len(sents), max_length), dtype='int32') * word_index[PAD]
        for i, sent in enumerate(sents):
            for j, word in enumerate(sent):
                if j >= max_length:
                    break
                Xs[i, j] = word_index.get(word, oov_idx)
        ys = model.predict(Xs)
        text_score[text] = ys
    return text_score


def get_model_dir(model_name, fold):
    return os.path.join(MODEL_DIR, '%s.fold%d' % (model_name, fold))
