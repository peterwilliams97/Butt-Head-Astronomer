import numpy as np
from keras.callbacks import EarlyStopping
import os
import time
import random
from framework import MODEL_DIR, LABEL_COLS, df_to_sentences, train_test_split
from utils import (dim, xprint, RocAucEvaluation, SaveAllEpochs, save_model, load_model,
    save_json, load_json)
from token_spacy2 import SpacySentenceTokenizer
from lstm import build_lstm, compile_lstm
from reductions import reduce, MEAN


OOV = '[OOV]'
PAD = '[PAD]'
tokenizer = None


class ClfVector:

    def __init__(self, n_hidden=64, max_length=100, max_features=20000,  # Shape
        dropout=0.5, learn_rate=0.001, learn_rate_unfrozen=0.001, frozen=False,  # General NN config
        epochs=5, epochs2=40, batch_size=100,
        lstm_type=1, predict_method=MEAN, token_method=None,
        single_oov=False, do_spacy=False, randomized=False,
        force_fit=False):
        """
            n_hidden: Number of elements in the LSTM layer
            max_length: Max length of comment text
            max_features: Maximum vocabulary size

            frozen => freeze embeddings
            2 stages of training: frozen, unfrozen

        """
        global tokenizer
        self.n_hidden = n_hidden
        self.max_length = max_length
        self.dropout = dropout
        self.learn_rate = learn_rate
        self.learn_rate_unfrozen = learn_rate_unfrozen
        self.max_features = max_features
        self.epochs = epochs
        self.epochs2 = epochs2
        self.batch_size = batch_size
        self.lstm_type = lstm_type
        self.token_method = token_method
        self.single_oov = single_oov
        self.do_spacy = False
        self.randomized = randomized

        D = self.__dict__
        model_name = 'pipe.%s' % '-'.join(str(D[k]) for k in sorted(D))

        # Predict method goes in description only
        self.predict_method = predict_method
        description = ', '.join('%s=%s' % (k, v) for k, v in sorted(D.items()))

        # xprint('model_name=%s' % model_name)
        self.description = description
        self.model_name = model_name
        xprint('model_name=%s' % self.model_name)
        # assert False

        # Don't include these parameters in description
        self.force_fit = force_fit
        self._shown_paths = False

        self.best_epoch = -1
        assert lstm_type in build_lstm, self.description

        tokenizer = SpacySentenceTokenizer(method=self.token_method)

    def __repr__(self):
        return 'ClfVector(%s)' % self.description

    def _get_paths(self, create_dir):
        model_dir = get_model_dir(self.model_name, 0)
        if create_dir:
            os.makedirs(model_dir, exist_ok=True)
        # RocAucEvaluation saves the trainable part of the model
        model_path = os.path.join(model_dir, 'model')
        config_path = os.path.join(model_dir, 'config.json')
        word_path = os.path.join(model_dir, 'words.json')
        epoch_path = os.path.join(model_dir, 'epochs.json')
        epoch_dict = load_json(epoch_path, {})
        if not self._shown_paths:
            xprint('model_path=%s exists=%s' % (model_path, os.path.exists(model_path)))
            xprint('config_path=%s exists=%s' % (config_path, os.path.exists(config_path)))
            xprint('word_path=%s exists=%s' % (word_path, os.path.exists(word_path)))
            xprint('epoch_path=%s exists=%s' % (epoch_path, os.path.exists(epoch_path)))
            xprint('epoch_dict=%s' % epoch_dict)
            self._shown_paths = True
        return model_path, config_path, word_path, epoch_path, epoch_dict

    def fit(self, train, test_size=0.1):
        xprint('ClfVector.fit', '-' * 80)

        model_path, config_path, word_path, epoch_path, epoch_dict = self._get_paths(True)
        if not self.force_fit:
            if (os.path.exists(model_path) and os.path.exists(config_path) and
                os.path.exists(word_path) and
                epoch_dict.get('done2', False)):
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
        lstm, epoch_dict = do_fit(X_train, y_train, X_val, y_val, lstm_shape,
            lstm_settings, {},
            batch_size=self.batch_size, lstm_type=self.lstm_type,
            epochs1=self.epochs, epochs2=self.epochs2,
            model_path=model_path, config_path=config_path, word_path=word_path, epoch_path=epoch_path,
            max_features=self.max_features, single_oov=self.single_oov,
            randomized=self.randomized)

        assert isinstance(epoch_dict, dict), epoch_dict
        xprint('****: best_epochs=%s - %s ' % (epoch_dict, self.description))
        del lstm

    def predict_reductions(self, test, predict_methods):
        print('ClfVector.predict_reductions', '-' * 80)
        X_test = df_to_sentences(test)
        model_path, config_path, word_path, _, _ = self._get_paths(False)

        assert os.path.exists(model_path), model_path
        assert os.path.exists(config_path), config_path

        return predict_reductions(model_path, config_path, word_path, X_test,
            predict_methods, self.max_length)

    def predict(self, test):
        reductions = self.predict_reductions(test, predict_methods=[self.predict_method])
        print('ClfVector.predict', '-' * 80)
        return reductions[self.predict_method]


def do_fit(train_texts, train_labels, dev_texts, dev_labels, lstm_shape, lstm_settings,
    lstm_optimizer, batch_size=100, epochs1=5, epochs2=5,
    model_path=None, config_path=None, epoch_path=None, word_path=None,
    lstm_type=1, max_features=None, single_oov=None, do_spacy=False, randomized=False):
    """Train a Keras model on the sentences in `train_texts`
        All the sentences in a text have the text's label
    """
    max_length = lstm_shape['max_length']

    xprint('do_fit: train_texts=%s dev_texts=%s' % (dim(train_texts), dim(dev_texts)))

    X_sents, word_count = tokenizer.token_lists(train_texts, max_length)
    validation_data = None
    if dev_texts is not None:
        y_sents, _ = tokenizer.token_lists(dev_texts, max_length)
    tokenizer.flush()

    if do_spacy:
        embeddings, word_index = get_spacy_embeddings(max_features, word_count, single_oov)
    else:
        embeddings, word_index = get_fasttext_embeddings(max_features, word_count, randomized)
    assert 0 <= min(word_index.values()) and max(word_index.values()) < embeddings.shape[0]
    save_json(word_path, word_index)

    X_train, w_train, y_train = apply_word_index(max_length, X_sents, word_index, train_texts, train_labels, 'train')
    if dev_texts is not None:
        X_val, w_val, y_val = apply_word_index(max_length, y_sents, word_index, dev_texts, dev_labels, 'dev')
        validation_data = (X_val, y_val)

    print('^^^embeddings=%s' % dim(embeddings))
    print('^^^X_train=%d..%d' % (X_train.min(), X_train.max()))
    print('^^^w_train=%g..%g mean=%g' % (w_train.min(), w_train.max(), w_train.mean()))
    print('^^^X_val=%d..%d' % (X_val.min(), X_val.max()))
    print('^^^w_val=%g..%g mean=%g' % (w_val.min(), w_val.max(), w_val.mean()))
    assert 0 <= X_train.min() and X_train.max() < embeddings.shape[0]
    assert 0 <= X_val.min() and X_val.max() < embeddings.shape[0]

    model = build_lstm[lstm_type](embeddings, lstm_shape, lstm_settings)

    xprint('built and compiled models')

    param_list = [(lstm_settings['lr'], True, 'epoch1', 'done1', epochs1,),
                  (lstm_settings['lr_unfrozen'], False, 'epoch2', 'done2', epochs2)]
    best_epochs = {}

    for run, (learning_rate, frozen, epoch_key, done_key, epochs) in enumerate(param_list):
        xprint('do_fit: run=%d learning_rate=%g frozen=%s epoch_key=%s done_key=%s' % (run,
            learning_rate, frozen, epoch_key, done_key))

        epoch_dict = load_json(epoch_path, {})
        xprint('epoch_dict=%s' % epoch_dict)
        epochs -= epoch_dict.get(epoch_key, 0)
        if epoch_dict.get(done_key, False) or epochs <= 0:
            xprint('do_fit: run %d is complete')
            continue

        if run > 0:
            xprint('Reloading partially trained model')
            model = load_model(model_path, config_path)

        compile_lstm(model, learning_rate, frozen)
        callback_list = None

        if validation_data is not None:
            ra_val = RocAucEvaluation(validation_data=validation_data, w_val=w_val,
                interval=1, epoch_key=epoch_key,
                model_path=model_path, config_path=config_path, epoch_path=epoch_path, do_prime=run > 0)
            early = EarlyStopping(monitor='val_auc', mode='max', patience=2, verbose=1)
            callback_list = [ra_val, early]
        else:
            sae = SaveAllEpochs(model_path, config_path, epoch_path, True)
            callback_list = [sae]

        print('!^^^embeddings=%s' % dim(embeddings))
        print('!^^^X_train=%d..%d' % (X_train.min(), X_train.max()))
        print('!^^^X_val=%d..%d' % (X_val.min(), X_val.max()))

        model.fit(X_train, y_train, sample_weight=w_train,
                  batch_size=batch_size, epochs=epochs, callbacks=callback_list,
                  validation_data=validation_data, verbose=1)
        if validation_data is not None:
            best_epochs[run] = ra_val.best_epoch
            ra_val.best_epoch = -1
        else:
            save_model(model, model_path, config_path)

        epoch_dict = load_json(epoch_path, {})
        epoch_dict[done_key] = True
        save_json(epoch_path, epoch_dict)
        xprint('do_fit: run=%d epoch_dict=%s' % (run, epoch_dict))

    return model, epoch_dict


def apply_word_index(max_length, sent_texts, word_index, texts_in, labels_in, name):
    n_sentences = sum(len(sents) for sents in sent_texts)

    # PAD = 0 so Xs is all PADs to start with
    Xs = np.zeros((n_sentences, max_length), dtype='int32')
    weights = np.zeros(n_sentences, dtype=np.float32)
    ys = np.zeros((n_sentences, len(LABEL_COLS)), dtype='int32')
    xprint('extract_sentences: Xs=%s ys=%s' % (dim(Xs), dim(ys)))

    order = list(range(n_sentences))  # !@#$ Try removing
    random.shuffle(order)
    oov_idx = word_index[OOV]

    t0 = time.perf_counter()
    N = max(10000, n_sentences // 5)
    # assert N > 1000, (n_sentences, N)
    i = 0
    for sents, y in zip(sent_texts, labels_in):
        wgt = 1.0 / len(sents)
        for sent in sents:
            for j, word in enumerate(sent[:max_length]):
                Xs[order[i], j] = word_index.get(word, oov_idx)
            weights[order[i]] = wgt
            ys[order[i]] = y
            if i % N == 0 or (i + 1) == n_sentences:
                dt = max(time.perf_counter() - t0, 1.0)
                if dt > 2.0 or (i + 1) == n_sentences:
                    print('~~~%5s %7d (%5.1f%%) sents dt=%4.1f sec %3.1f sents/sec' %
                         (name, i, 100.0 * i / n_sentences, dt, i / dt))
            i += 1

    weights /= weights.mean()
    return Xs, weights, ys


# max_features = 30000
# maxlen = 100
embed_size = 300

EMBEDDING_FILE = '~/data/models/fasttext/crawl-300d-2M.vec'
EMBEDDING_FILE = os.path.expanduser(EMBEDDING_FILE)

# from embeddings import local_path
# embeddings_local = local_path(EMBEDDING_FILE)


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


embeddings_index = None
emb_mean, emb_std = None, None


def get_fasttext_embeddings(max_features, word_count, randomized):
    global embeddings_index, emb_mean, emb_std

    if embeddings_index is None:
        embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))
        all_embs = np.stack(embeddings_index.values())
        xprint('all_embs=%s' % dim(all_embs))
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        xprint('emb_mean, emb_std=%s' % [emb_mean, emb_std])

    word_list = sorted(word_count, key=lambda w: (-word_count[w], w))
    vocab = [OOV, PAD] + word_list
    vocab = vocab[:max_features]
    word_index = {w: i for i, w in enumerate(vocab)}

    nb_words = min(max_features, len(word_index))
    print('nb_words=%d randomized=%s' % (nb_words, randomized))
    if randomized:
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    else:
        embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    print('embedding_matrix=%s' % dim(embedding_matrix))
    return embedding_matrix, word_index


def get_spacy_embeddings(max_features, word_count, single_oov):
    """Returns: embedding matrix n_words x embed_size
                n_words <= max_features
    """
    # Use these vectors to create our embedding matrix, with random initialization for words
    # that aren't in GloVe. We'll use the same mean and stdev of embeddings the GloVe has when
    # generating the random init.

    assert isinstance(single_oov, bool), single_oov

    nlp = tokenizer._load_nlp()
    vectors = nlp.vocab.vectors
    data = nlp.vocab.vectors.data
    embed_size = data.shape[1]
    emb_mean, emb_std = data.mean(), data.std()

    xprint('get_spacy_embeddings: max_features=%d word_count=%d single_oov=%s' % (max_features,
        len(word_count), single_oov))
    xprint('get_spacy_embeddings: data=%s emb_mean=%.3f emb_std=%.3f range=%.3f %.3f' % (dim(data),
        emb_mean, emb_std, data.min(), data.max()))

    word_list = sorted(word_count, key=lambda w: (-word_count[w], w))
    word_vid = {w: vectors.find(key=w) for w in word_list}
    if single_oov:
        word_list = [w for w in word_list if word_vid[w] >= 0]
    vocab = [OOV, PAD] + word_list
    vocab = vocab[:max_features]
    oov = [word for word in word_list if word_vid[word] < 0]
    assert vocab, (len(word_list), len(word_vid), word_list[:20], sorted(word_vid)[:20])

    embeddings = np.random.normal(emb_mean, emb_std, (len(vocab), embed_size)).astype(np.float32)
    xprint('get_spacy_embeddings: embeddings=%s mean=%.3f std=%.3f range=%.3f %.3f' % (dim(embeddings),
        embeddings.mean(), embeddings.std(), embeddings.min(), embeddings.max()))
    embeddings[1, :] = np.zeros(embed_size, dtype=np.float32)  # PAD
    for i, word in enumerate(vocab[2:]):
        w_vid = word_vid[word]
        if w_vid >= 0:
            embeddings[i:, :] = data[w_vid, :]

    xprint('get_spacy_embeddings: embeddings=%s mean=%.3f std=%.3f range=%.3f %.3f' % (dim(embeddings),
        embeddings.mean(), embeddings.std(), embeddings.min(), embeddings.max()))

    xprint('get_spacy_embeddings: oov=%d=%.3f %s' % (len(oov), len(oov) / len(word_count), oov[:50]))

    xprint('get_spacy_embeddings:'
           '\n max_features=%d word_list=%d word_index=%d data=%s'
           '\n -> embeddings=%s vocab=%d=%.3f'
        % (max_features, len(word_list), len(word_vid), dim(data),
           dim(embeddings), len(vocab), len(vocab) / len(word_list)))

    word_index = {w: i for i, w in enumerate(vocab)}
    xprint('get_spacy_embeddings: word_index= %d - %d (%d values)' % (
        min(word_index.values()),
        max(word_index.values()),
        len(word_index)))
    assert 0 <= min(word_index.values()) and max(word_index.values()) < embeddings.shape[0]
    return embeddings, word_index


def compute_reduction_weights(model_path, config_path, word_path, texts, methods, max_length):
    text_ys = predict_ys(model_path, config_path, word_path, texts, max_length)

    reductions = {method: np.zeros((len(texts), len(LABEL_COLS)), dtype=np.float32)
                  for method in methods}
    for i, text in enumerate(texts):
        ys = text_ys[text]
        for method in methods:
            reductions[method][i, :] = reduce(ys, method)

    return reductions


def predict_reductions(model_path, config_path, word_path, texts, methods, max_length):
    text_ys = predict_ys(model_path, config_path, word_path, texts, max_length)

    reductions = {method: np.zeros((len(texts), len(LABEL_COLS)), dtype=np.float32)
                  for method in methods}
    for i, text in enumerate(texts):
        ys = text_ys[text]
        for method in methods:
            reductions[method][i, :] = reduce(ys, method)

    return reductions


def predict_ys(model_path, config_path, word_path, texts_in, max_length):
    xprint('predict_reductions(model_path=%s, config_path=%s, word_path=%s, texts=%s, max_length=%d)'
           % (model_path, config_path, word_path, dim(texts_in), max_length))

    model = load_model(model_path, config_path)
    word_index = load_json(word_path)
    oov_idx = word_index[OOV]

    sents_list, word_count = tokenizer.token_lists(texts_in, max_length)
    assert len(texts_in) == len(sents_list), (len(texts_in), len(sents_list))

    n_sentences = sum(len(sents) for sents in sents_list)
    t0 = time.perf_counter()
    N = max(1000, n_sentences // 5)

    text_ys = {}
    n = 0
    for text, sents in zip(texts_in, sents_list):
        Xs = np.ones((len(sents), max_length), dtype='int32') * word_index[PAD]
        for i, sent in enumerate(sents):
            for j, word in enumerate(sent):
                if j >= max_length:
                    break
                Xs[i, j] = word_index.get(word, oov_idx)
            if n % N == 0 or (n + 1) == n_sentences:
                dt = max(time.perf_counter() - t0, 1.0)
                if dt > 2.0 or (n + 1) == n_sentences:
                    print('``` %7d (%5.1f%%) sents dt=%4.1f sec %3.1f sents/sec' %
                         (n, 100.0 * n / n_sentences, dt, n / dt))
            n += 1
        ys = model.predict(Xs)
        text_ys[text] = ys

    return text_ys


def get_model_dir(model_name, fold):
    return os.path.join(MODEL_DIR, '%s.fold%d' % (model_name, fold))
