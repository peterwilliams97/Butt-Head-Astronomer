import os
import random
import time
from keras.layers import GRU, LSTM
from utils import xprint, xprint_init
from gru_framework import (Evaluator, set_n_samples, set_random_seed, show_results,
    get_n_samples_str, SUMMARY_DIR)
from clf_rec_word_char import ClfRecWordChar
from clf_rec_word import ClfRecWord


def get_clf_word(max_features, maxlen, dropout, n_hidden, Rec, batch_size, epochs):
    return ClfRecWord(max_features=max_features, maxlen=maxlen, dropout=dropout, n_hidden=n_hidden,
        Rec=Rec, trainable=trainable, batch_size=batch_size, epochs=epochs)


def get_clf_word_char(max_features, maxlen, dropout, n_hidden, Rec, batch_size, epochs):
    return ClfRecWordChar(max_features=max_features, maxlen=maxlen, dropout=dropout, n_hidden=n_hidden,
        Rec=Rec, trainable=trainable, batch_size=batch_size, epochs=epochs)


set_n_samples(40000)
submission_name = 'ft_explore_1'
xprint_init(submission_name, False)
auc_list = []
run_summary_path = os.path.join(SUMMARY_DIR,
    '%s.%s.run_summary.json' % (submission_name, get_n_samples_str()))
xprint('run_summary_path=%s' % run_summary_path)

# Set params
max_features = 150000
maxlen = 150
epochs = 40
batch_size = 128
trainable = False

# params = (maxlen, max_features, n_hidden, dropout, batch_size)
params0 = (128, 0.2)

params_list = []
for n_hidden in [100, 128, 150, 200]:
    for dropout in [0.2, 0.3, 0.5]:  # [0.1, 0.3, 0.5]:
        params = (n_hidden, dropout)
        params_list.append(params)

xprint('params_list=%d' % len(params_list))
random.seed(time.time())
random.shuffle(params_list)
params_list = [params for params in params_list if params != params0]
assert len(params0) == len(params_list[0])
params_list = [params0] + params_list

xprint('params_list=%d' % len(params_list))
for i, params in enumerate(params_list[:10]):
    print(i, params)
xprint('$' * 100)

for n_runs0 in range(2):
    xprint('@' * 100)
    xprint('n_runs0=%d' % n_runs0)

    p_i = 0
    for params in params_list:
        n_hidden, dropout = params
        for Rec in [GRU, LSTM]:
            for get_clf in [get_clf_word, get_clf_word_char]:

                xprint('#' * 100)
                xprint('n_runs0=%d p_i=%d of %d' % (n_runs0, p_i, len(params_list) * 4))
                xprint('params=%s %s %s' % (get_clf.__name__, Rec.__name__, list(params)))
                set_random_seed(10000 + n_runs0)
                evaluator = Evaluator(frac=.5)
                auc, best_epoch, best_auc, dt_fit, dt_pred = evaluator.evaluate(get_clf,
                    max_features, maxlen, dropout, n_hidden, Rec, batch_size, epochs)

                auc_list.append((auc, best_epoch, best_auc, dt_fit, dt_pred,
                    get_clf.__name__, str(evaluator.clf_)))
                show_results(auc_list)
                xprint('&' * 100)
                p_i += 1

xprint('$' * 100)
