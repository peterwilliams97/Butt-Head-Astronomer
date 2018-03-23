import random
import time
from keras.layers import GRU, LSTM
from utils import xprint, xprint_init
from functools import partial
from gru_framework import (set_n_samples, get_n_samples_str, prepare_data, CV_predictor)
from mod_rec_word_char import ClfRecWordChar
from mod_rec_word import ClfRecWord


def get_clf_word(max_features, maxlen, dropout, n_hidden, Rec, batch_size):
    return ClfRecWord(max_features=max_features, maxlen=maxlen, dropout=dropout, n_hidden=n_hidden,
        Rec=Rec, batch_size=batch_size, epochs=1)


def get_clf_word_char1(max_features, maxlen, dropout, n_hidden, Rec, batch_size):
    return ClfRecWordChar(max_features=max_features, maxlen=maxlen, dropout=dropout, n_hidden=n_hidden,
        Rec=Rec, batch_size=batch_size, epochs=1, model_type=1)


def get_clf_word_char2(max_features, maxlen, dropout, n_hidden, Rec, batch_size):
    return ClfRecWordChar(max_features=max_features, maxlen=maxlen, dropout=dropout, n_hidden=n_hidden,
        Rec=Rec, batch_size=batch_size, epochs=1, model_type=2)


set_n_samples(40000)
submission_name = 'ft_cv_explore_3'
xprint_init(submission_name, False)
auc_list = []

# Set params
epochs = 2
batch_size = 200

params_list = []
for maxlen in [10, 150]:  # [50, 75, 100, 150]:
    for max_features in [100, 50000, 40000]:  # [20000, 25000, 30000, 40000]:
        for n_hidden in [200, 150]:
            for dropout in [0.5, 0.3]:  # [0.1, 0.3, 0.5]:
                params = (maxlen, max_features, n_hidden, dropout)
                params_list.append(params)

xprint('params_list=%d' % len(params_list))
params0, params_list1 = params_list[0], params_list[1:]
random.seed(time.time())
random.shuffle(params_list1)
assert len(params0) == len(params_list1[0])
params_list = [params0] + params_list1

xprint('params_list=%d' % len(params_list))
for i, params in enumerate(params_list[:10]):
    print(i, params)
xprint('$' * 100)

x_train, y_train, x_test = prepare_data()
n_splits = 2

p_i = 0
for params in params_list:
    maxlen, max_features, n_hidden, dropout = params
    for Rec in [GRU, LSTM]:
        for get_clf_base in [ get_clf_word_char1, get_clf_word, get_clf_word_char2,]:
            get_clf = partial(get_clf_base, max_features, maxlen, dropout, n_hidden, Rec, batch_size)

            xprint('#' * 100)
            xprint('p_i=%d of %d' % (p_i, len(params_list) * 4))
            xprint('params=%s %s %s' % (get_clf_base.__name__, Rec.__name__, list(params)))
            evaluator = CV_predictor(get_clf, x_train, y_train, x_test, n_splits, batch_size, epochs)
            evaluator.predict()
            xprint('&' * 100)
            p_i += 1

xprint('$' * 100)