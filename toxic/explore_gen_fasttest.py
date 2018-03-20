import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback, EarlyStopping
import os
import random
import time
from utils import dim, xprint, xprint_init, load_json, save_json, load_model, save_model
from gru_framework import (Evaluator, set_n_samples, set_random_seed, show_results,
    get_n_samples_str, auc_score_list, SUMMARY_DIR, X_train0, X_test0)


HACK = False
np.random.seed(42)

os.environ['OMP_NUM_THREADS'] = '4'


EMBEDDING_FILE = '~/data/models/fasttext/crawl-300d-2M.vec'
# TRAIN = '~/data/toxic/train.csv'
# TEST = '~/data/toxic/test.csv'
# SUBMISSION = '~/data/toxic/sample_submission.csv'

EMBEDDING_FILE = os.path.expanduser(EMBEDDING_FILE)
# TRAIN = os.path.expanduser(TRAIN)
# TEST = os.path.expanduser(TEST)
# SUBMISSION = os.path.expanduser(SUBMISSION)

# train = pd.read_csv(TRAIN)
# test = pd.read_csv(TEST)
# submission = pd.read_csv(SUBMISSION)

EMBEDDING_FILE = os.path.expanduser(EMBEDDING_FILE)
assert os.path.exists(EMBEDDING_FILE)

# X_train0 = train["comment_text"].fillna("fillna").values
# y_train0 = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
# X_test0 = test["comment_text"].fillna("fillna").values


# max_features = 30000
# maxlen = 100
embed_size = 300
char_embed_size = 25

char_index_map = {}


def get_char_index(max_features, char_maxlen):
    global char_index_map
    if maxlen not in char_index_map:
        char_count = defaultdict(int)
        for doc in X_train0:
            for c in doc[:char_maxlen]:
                char_count[c] += 1
        char_list = sorted(char_count, key=lambda c: (-char_count[c], c))
        char_list = ['UNK'] + char_list
        char_list = char_list[:max_features]
        char_index_map[maxlen] = {c: i for i, c in enumerate(char_list)}
    print('*** char_index=%d' % len(char_index_map[maxlen]))
    return char_index_map[maxlen]


tokenizer_map = {}


def get_tokenizer(max_features, maxlen):
    global tokenizer_map
    if (max_features, maxlen) not in tokenizer_map:
        tokenizer = text.Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(list(X_train0) + list(X_test0))
        tokenizer_map[(max_features, maxlen)] = tokenizer
    return tokenizer_map[(max_features, maxlen)]


def tokenize(max_features, maxlen, X):
    tokenizer = get_tokenizer(max_features, maxlen)
    X = tokenizer.texts_to_sequences(X)
    X = sequence.pad_sequences(X, maxlen=maxlen)
    return X


def char_tokenize(max_features, maxlen, char_maxlen, X):
    char_index = get_char_index(max_features, maxlen)
    unk_idx = char_index['UNK']
    X_char = np.zeros((X.shape[0], char_maxlen), dtype=np.int32)
    for i in range(X.shape[0]):
        for j, c in enumerate(X[i][:char_maxlen]):
            X_char[i, j] = char_index.get(c, unk_idx)
    return X_char


def tokenize_all(max_features, maxlen):
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X_train0) + list(X_test0))
    X_train = tokenizer.texts_to_sequences(X_train0)
    X_test = tokenizer.texts_to_sequences(X_test0)
    x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    return tokenizer.word_index, x_train, x_test


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


embeddings_index = None


def get_embeddings(max_features, maxlen):
    global embeddings_index

    print('get_embeddings: embeddings_index=%s' % (embeddings_index is not None))

    if embeddings_index is None:
        embeddings_index = {}
        with open(EMBEDDING_FILE) as f:
            for i, o in enumerate(f):
                w, vec = get_coefs(*o.rstrip().rsplit(' '))
                embeddings_index[w] = vec
                if HACK:
                    if i > 11000:  # !@#$
                        break
        # embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

    tokenizer = get_tokenizer(max_features, maxlen)
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    xprint('embedding_matrix=%s' % dim(embedding_matrix))
    return embedding_matrix


char_embeddings_index = None


def get_char_embeddings(max_features, maxlen):
    global char_embeddings_index

    print('get_char_embeddings: char_embeddings_index=%s' % (char_embeddings_index is not None))

    char_index = get_char_index(max_features, maxlen)
    n_chars = min(max_features, len(char_index))
    print('get_char_embeddings: n_chars=%d max_features=%d char_index=%d' % (
        n_chars, max_features, len(char_index)))
    # embedding_matrix = np.zeros((n_chars, char_embed_size))
    embedding_matrix = np.random.normal(0.0, 1.0, (n_chars, char_embed_size))
    embedding_matrix = np.clip(embedding_matrix, -1.0, 1.0)
    xprint('embedding_matrix=%s' % dim(embedding_matrix))
    return embedding_matrix


model_path = 'gruxx_model.pkl'
config_path = 'gruxx_model.json'


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.X_char_val, self.y_val = validation_data
        self.best_epoch = -1
        self.best_auc = 0.0

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval != 0:
            return

        print('!!!!!')
        # y_pred = self.model.predict(self.X_val, verbose=0)
        y_pred = self.model.predict(x={'w_inp': self.X_val, 'c_inp': self.X_char_val},
            batch_size=2000, verbose=0)

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


def get_model(embedding_matrix, char_embedding_matrix,
    max_features, maxlen, char_maxlen, dropout=0.2, n_hidden=80):

    print('embedding_matrix=%s char_embedding_matrix=%s' % (type(embedding_matrix),
          type(char_embedding_matrix)))
    print('max_features=%s maxlen=%s char_maxlen=%s' % (type(max_features),
          type(maxlen), type(char_maxlen)))
    assert isinstance(max_features, int)
    assert isinstance(maxlen, int)
    assert isinstance(char_maxlen, int)

    inp1 = Input(shape=(maxlen, ), name='w_inp')
    x = Embedding(embedding_matrix.shape[0],
                  embedding_matrix.shape[1],
                  weights=[embedding_matrix], name='w_emb')(inp1)
    x = SpatialDropout1D(dropout, name='w_drop')(x)

    x = Bidirectional(GRU(n_hidden, return_sequences=True), name='w_bidi')(x)
    avg_pool = GlobalAveragePooling1D(name='w_ave')(x)
    max_pool = GlobalMaxPooling1D(name='w_max')(x)
    conc1 = concatenate([avg_pool, max_pool], name='w_conc')
    # conc1 = Bidirectional(GRU(n_hidden, return_sequences=False))(x)

    inp2 = Input(shape=(char_maxlen, ), name='c_inp')
    x = Embedding(char_embedding_matrix.shape[0],
                  char_embedding_matrix.shape[1],
                  weights=[char_embedding_matrix], name='c_emb')(inp2)
    x = SpatialDropout1D(dropout, name='c_drop')(x)

    # x = Bidirectional(LSTM(n_hidden, return_sequences=True))(x)
    # avg_pool = GlobalAveragePooling1D()(x)
    # max_pool = GlobalMaxPooling1D()(x)
    # conc2 = concatenate([avg_pool, max_pool])
    conc2 = Bidirectional(GRU(n_hidden, return_sequences=False), name='c_bidi')(x)

    conc = concatenate([conc1, conc2], name='w_c_conc')

    outp = Dense(6, activation="sigmoid", name='output')(conc)

    model = Model(inputs=[inp1, inp2], outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    return model


class ClfGru():

    def __init__(self, max_features=30000, char_max_features=1000, maxlen=100,
        dropout=0.2, n_hidden=80, batch_size=32,
        epochs=2, validate=True):
        self.maxlen = maxlen
        self.char_maxlen = maxlen * 4
        self.max_features = max_features
        self.char_max_features = char_max_features
        self.dropout = dropout
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.epochs = epochs
        self.validate = validate

        D = self.__dict__
        self.description = ', '.join('%s=%s' % (k, v) for k, v in sorted(D.items()))

        self.embedding_matrix = get_embeddings(self.max_features, self.maxlen)
        self.char_embedding_matrix = get_char_embeddings(self.char_max_features, self.char_maxlen)
        # self.model = get_model(self.embedding_matrix, max_features, maxlen, dropout, n_hidden)
        self.model = get_model(self.embedding_matrix, self.char_embedding_matrix,
              self.max_features, self.maxlen, self.maxlen * 4,
              dropout=self.dropout, n_hidden=self.n_hidden)

    def __repr__(self):
        return 'ClfGru(%s)' % self.description

    def fit(self, X_train_in, y_train):
        X_train = tokenize(self.max_features, self.maxlen, X_train_in)
        X_char_train = char_tokenize(self.char_max_features, self.maxlen, self.char_maxlen, X_train_in)

        print('X_char_train=%s' % dim(X_char_train))
        print('X_char_train=%d %d' % (X_char_train.min(), X_char_train.max()))
        assert np.all(X_char_train <= self.max_features)

        if self.validate:
            X_train, X_val, X_char_train, X_char_val, y_train, y_val = train_test_split(
                X_train, X_char_train, y_train, train_size=0.95, random_state=233)

            RocAuc = RocAucEvaluation(validation_data=(X_val, X_char_val, y_val), interval=1)
            early = EarlyStopping(monitor='val_auc', mode='max', patience=2, verbose=1)
            callback_list = [RocAuc, early]

            self.model.fit(x={"w_inp": X_train, "c_inp": X_char_train}, y=y_train,
                           validation_data=({"w_inp": X_val, "c_inp": X_char_val}, y_val),
                           epochs=self.epochs, callbacks=callback_list, verbose=1)

            xprint('***Best epoch=%d acu=%.4f' % (RocAuc.best_epoch + 1, RocAuc.best_auc))

    def predict(self, X_test_in):
        self.model = load_model(model_path, config_path)
        X_test = tokenize(self.max_features, self.maxlen, X_test_in)
        X_char_test = char_tokenize(self.max_features, self.maxlen, self.char_maxlen, X_test_in)

        # y_pred = self.model.predict(X_test, batch_size=2000)
        y_pred = self.model.predict(x={'w_inp': X_test, 'c_inp': X_char_test}, batch_size=2000)

        return y_pred


def get_clf():
    return ClfGru(max_features=max_features, maxlen=maxlen, dropout=dropout, n_hidden=n_hidden,
        batch_size=batch_size, epochs=epochs)


set_n_samples(20000)
# evaluator = Evaluator()
# evaluator.evaluate(get_clf)

# Original values
max_features = 30000
maxlen = 100
n_hidden = 80
dropout = 0.2
batch_size = 32
epochs = 40

# params = (maxlen, max_features, n_hidden, dropout, batch_size)
params0 = (27, 1000, 11, 0.2, 55)
params_list = []
for maxlen in [150]:  # [50, 75, 100, 150]:
    for max_features in [30000, 50000, 70000]:  # [20000, 25000, 30000, 40000]:
        for n_hidden in [150, 200]:
            for dropout in [0.2, 0.3, 0.5]:  # [0.1, 0.3, 0.5]:
                for batch_size in [32]:
                    params = (maxlen, max_features, n_hidden, dropout, batch_size)
                    params_list.append(params)

print('params_list=%d' % len(params_list))
random.seed(time.time())
random.shuffle(params_list)
# params_list = [params0] + params_list
# params_list.reverse()
print('params_list=%d' % len(params_list))
for i, params in enumerate(params_list[:10]):
    print(i, params)
print('$' * 100)

submission_name = 'ggru_general_1'
xprint_init(submission_name, False)
auc_list = []
run_summary_path = os.path.join(SUMMARY_DIR,
    '%s.%s.run_summary.json' % (submission_name, get_n_samples_str()))

completed_tests = load_json(run_summary_path, {})
xprint('run_summary_path=%s' % run_summary_path)
n_completed0 = len(completed_tests)

for n_runs0 in range(2):
    print('n_completed0=%d n_runs0=%d' % (n_completed0, n_runs0))

    for p_i, (maxlen, max_features, n_hidden, dropout, batch_size) in enumerate(params_list):
        # for Rec in [GRU, LSTM]:

            clf_str = str(get_clf())
            xprint('#' * 80)
            xprint('params %d: %s' % (p_i, clf_str))
            runs = completed_tests.get(clf_str, [])
            # if len(runs) > n_runs0:
            #     xprint('skipping runs=%d n_runs0=%d' % (len(runs), n_runs0))
            #     continue

            set_random_seed(10000 + n_runs0)
            evaluator = Evaluator()
            auc = evaluator.evaluate(get_clf)

            auc_list.append((auc, get_clf.__name__, str(get_clf())))
            show_results(auc_list)

            runs.append(auc_score_list(auc))
            completed_tests[str(get_clf())] = runs
            save_json(run_summary_path, completed_tests)
            xprint('n_completed=%d = %d + %d' % (len(completed_tests), n_completed0,
                len(completed_tests) - n_completed0))
            xprint('&' * 100)

xprint('$' * 100)
