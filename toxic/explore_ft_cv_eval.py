import random
import time
from utils import xprint, xprint_init
from functools import partial
from gru_framework import (set_n_samples, get_n_samples_str, prepare_data, CV_predictor,
    show_results_cv, K_GRU, K_LSTM, save_submission)
from mod_rec_word_char import ClfRecWordChar


def get_clf_word_char1(max_features, maxlen, dropout, n_hidden, Rec, rnn_layers, batch_size):
    return ClfRecWordChar(max_features=max_features, maxlen=maxlen, dropout=dropout, n_hidden=n_hidden,
        Rec=Rec, rnn_layers=rnn_layers, trainable=trainable, batch_size=batch_size, epochs=1, model_type=1)


def get_clf_word_char2(max_features, maxlen, dropout, n_hidden, Rec, rnn_layers, batch_size):
    return ClfRecWordChar(max_features=max_features, maxlen=maxlen, dropout=dropout, n_hidden=n_hidden,
        Rec=Rec, rnn_layers=rnn_layers, trainable=trainable, batch_size=batch_size, epochs=1, model_type=2)


set_n_samples(20000)
submission_base = 'ft_cv_explore_3y.%s' % get_n_samples_str()
xprint_init(submission_base, False)

# RESULTS SUMMARY: 3
# auc=0.9898   0: ClfRecWordChar(batch_size=200, dropout=0.5, epochs=1, max_features=150000, maxlen=150, model_type=1, n_hidden=64, rec=CuDNNGRU, rnn_layers=2, trainable=False, char_max_features=1000, char_maxlen=600)
# auc=0.9896   2: ClfRecWordChar(batch_size=200, dropout=0.5, epochs=1, max_features=150000, maxlen=150, model_type=1, n_hidden=64, rec=CuDNNGRU, rnn_layers=3, trainable=False, char_max_features=1000, char_maxlen=600)
# auc=0.9894   1: ClfRecWordChar(batch_size=200, dropout=0.5, epochs=1, max_features=150000, maxlen=150, model_type=1, n_hidden=64, rec=CuDNNGRU, rnn_layers=1, trainable=False, char_max_features=1000, char_maxlen=600)

# RESULTS SUMMARY: 6
# auc=0.9897   4: ClfRecWordChar(batch_size=200, dropout=0.5, epochs=1, max_features=150000, maxlen=150, model_type=1, n_hidden=128, rec=CuDNNGRU, trainable=False, char_max_features=1000, char_maxlen=600)
# auc=0.9896   5: ClfRecWordChar(batch_size=200, dropout=0.5, epochs=1, max_features=150000, maxlen=150, model_type=1, n_hidden=128, rec=CuDNNLSTM, trainable=False, char_max_features=1000, char_maxlen=600)

# auc=0.9886   7: get_clf_word_char ClfRecWordChar(batch_size=128, dropout=0.3, epochs=40, max_features=150000, maxlen=150, n_hidden=128, rec=LSTM, trainable=False, validate=True, char_max_features=1000, char_maxlen=600) best_epoch=7 best_auc=0.9827 dt_fit=12331.1 sec dt_pred=26.5 sec
# auc=0.9882   5: get_clf_word_char ClfRecWordChar(batch_size=128, dropout=0.3, epochs=40, max_features=150000, maxlen=150, n_hidden=128, rec=GRU, trainable=False, validate=True, char_max_features=1000, char_maxlen=600) best_epoch=7
# Set params
epochs = 1  # 7
batch_size = 200
max_features = 150000
maxlen = 150
trainable = False

params_list = []
for n_hidden in [64, 128, 200]:
    for dropout in [0.5, 0.3]:
        params = (n_hidden, dropout)
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

(idx_train, X_train, y_train), (idx_test, X_test, y_test) = prepare_data(0.4)
n_splits = 4

auc_list = []
for params in params_list:
    n_hidden, dropout = params
    for get_clf_base in [get_clf_word_char1, get_clf_word_char2]:
        Rec = K_GRU
        for rnn_layers in [2, 1, 3]:
            n = len(auc_list)
            get_clf = partial(get_clf_base, max_features, maxlen, dropout, n_hidden, Rec,
                rnn_layers, batch_size)

            xprint('#' * 100)
            xprint('config=%d of %d' % (n, len(params_list) * 4))
            xprint('params=%s %s %s' % (get_clf_base.__name__, Rec.__name__, list(params)))
            evaluator = CV_predictor(get_clf, idx_train, X_train, y_train, idx_test, X_test,
                n_splits, batch_size, epochs)
            evaluator.predict()
            auc_list.append((evaluator.auc_train, str(get_clf())))
            show_results_cv(auc_list)
            evaluator.eval_predictions()
            # submission_name = '%s.%03d.csv' % (submission_base, n)
            # save_submission(evaluator.test_predictions, submission_name)
            xprint('&' * 100)

xprint('$' * 100)
