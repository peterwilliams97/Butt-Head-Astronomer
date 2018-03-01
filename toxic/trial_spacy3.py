# coding: utf-8
"""
    SpaCy deep_learning_keras.py solution to Kaggle Toxic Comment challenge
"""
import os
from utils import xprint_init, xprint, load_json, save_json
from framework import (Evaluator, set_random_seed, show_auc, set_n_samples, get_n_samples_str,
    auc_score_list)
from clf_spacy import ClfSpacy, PREDICT_METHODS


submission_name = 'spacy_lstm15'
epochs = 40
set_n_samples(10000)
SUMMARY_DIR = 'run.summaries'
os.makedirs(SUMMARY_DIR, exist_ok=True)
run_summary_path = os.path.join(SUMMARY_DIR,
    '%s.%s.run_summary.json' % (submission_name, get_n_samples_str()))


def get_clfx():
    return ClfSpacy(n_hidden=16, max_length=10,  # Shape
                    dropout=0.5, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=150, frozen=frozen,
                    lstm_type=lstm_type, predict_method=predict_method)


def get_clf0():
    return ClfSpacy(n_hidden=64, max_length=100,  # Shape
                    dropout=0.5, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=150, frozen=frozen,
                    lstm_type=lstm_type, predict_method=predict_method)


def get_clf1():
    return ClfSpacy(n_hidden=128, max_length=100,  # Shape
                    dropout=0.5, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=150, frozen=frozen,
                    lstm_type=lstm_type, predict_method=predict_method)


def get_clf2():
    return ClfSpacy(n_hidden=64, max_length=150,  # Shape
                    dropout=0.5, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=150, frozen=frozen,
                    lstm_type=lstm_type, predict_method=predict_method)


def get_clf3():
    return ClfSpacy(n_hidden=64, max_length=100,  # Shape
                    dropout=0.2, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=150, frozen=frozen,
                    lstm_type=lstm_type, predict_method=predict_method)


def get_clf4():
    return ClfSpacy(n_hidden=64, max_length=100,  # Shape
                    dropout=0.5, learn_rate=0.002,  # General NN config
                    epochs=epochs, batch_size=150, frozen=frozen,
                    lstm_type=lstm_type, predict_method=predict_method)


def get_clf5():
    return ClfSpacy(n_hidden=64, max_length=100,  # Shape
                    dropout=0.5, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=50, frozen=frozen,
                    lstm_type=lstm_type, predict_method=predict_method)


def get_clf6():
    return ClfSpacy(n_hidden=64, max_length=50,  # Shape
                    dropout=0.5, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=50, frozen=frozen,
                    lstm_type=lstm_type, predict_method=predict_method)


def get_clf12():
    return ClfSpacy(n_hidden=256, max_length=100,  # Shape
                    dropout=0.5, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=150, frozen=frozen,
                    lstm_type=lstm_type, predict_method=predict_method)


def get_clf13():
    return ClfSpacy(n_hidden=128, max_length=100,  # Shape
                    dropout=0.5, learn_rate=0.002,  # General NN config
                    epochs=epochs, batch_size=150, frozen=frozen,
                    lstm_type=lstm_type, predict_method=predict_method)


def get_clf14():
    return ClfSpacy(n_hidden=128, max_length=100,  # Shape
                    dropout=0.5, learn_rate=0.005,  # General NN config
                    epochs=epochs, batch_size=150, frozen=frozen,
                    lstm_type=lstm_type, predict_method=predict_method)


def get_clf15():
    return ClfSpacy(n_hidden=128, max_length=100,  # Shape
                    dropout=0.5, learn_rate=0.010,  # General NN config
                    epochs=epochs, batch_size=150, frozen=frozen,
                    lstm_type=lstm_type, predict_method=predict_method)


def show_results(auc_list):
    results = [(i, auc, clf, clf_str) for i, (auc, clf, clf_str) in enumerate(auc_list)]
    results.sort(key=lambda x: (-x[1].mean(), x[2], x[3]))
    xprint('~' * 100)
    xprint('RESULTS SO FAR: %d' % len(results))
    for i, auc, clf, clf_str in results:
        xprint('$' * 100)
        xprint('auc=%.4f %3d: %s %s' % (auc.mean(), i, clf, clf_str))
        show_auc(auc)
    xprint('^' * 100)
    xprint('RESULTS SUMMARY: %d' % len(results))
    for i, auc, clf, clf_str in results:
        xprint('auc=%.4f %3d: %s %s' % (auc.mean(), i, clf, clf_str))


# spacy_lstm15.log instance3 has
# RESULTS SUMMARY: 63
# auc=0.9919  46: get_clf12 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=2, max_length=100, n_hidden=256, predict_method=LINEAR)
# auc=0.9899  49: get_clf12 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=2, max_length=100, n_hidden=256, predict_method=LINEAR4)
# auc=0.9895  22: get_clf12 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=6, max_length=100, n_hidden=256, predict_method=LINEAR)
# auc=0.9885  34: get_clf12 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=5, max_length=100, n_hidden=256, predict_method=LINEAR)
# auc=0.9885  42: get_clf12 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=2, max_length=100, n_hidden=256, predict_method=MEAN_MAX)

# spacy_lstm15.log instance4 has
# RESULTS SUMMARY: 65
# auc=0.9900  34: get_clf1 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=6, max_length=100, n_hidden=128, predict_method=LINEAR2)
# auc=0.9897  60: get_clf1 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=2, max_length=100, n_hidden=128, predict_method=LINEAR2)
# auc=0.9877  62: get_clf1 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=2, max_length=100, n_hidden=128, predict_method=LINEAR4)

# spacy_lstm12.log instance4 has
# RESULTS SUMMARY: 37
# auc=0.9899  34: get_clf13 ClfSpacy(dropout=0.5, learn_rate=0.002, lstm_type=6, max_length=100, n_hidden=128, predict_method=LINEAR2)
# auc=0.9866  36: get_clf13 ClfSpacy(dropout=0.5, learn_rate=0.002, lstm_type=6, max_length=100, n_hidden=128, predict_method=LINEAR4)
# auc=0.9861  35: get_clf13 ClfSpacy(dropout=0.5, learn_rate=0.002, lstm_type=6, max_length=100, n_hidden=128, predict_method=LINEAR3)


# spacy_lstm14.log instance4 has
# RESULTS SUMMARY: 61
# auc=0.9815   7: get_clf13 ClfSpacy(dropout=0.5, learn_rate=0.002, lstm_type=8, max_length=100, n_hidden=128, predict_method=LINEAR)
# auc=0.9786   8: get_clf13 ClfSpacy(dropout=0.5, learn_rate=0.002, lstm_type=8, max_length=100, n_hidden=128, predict_method=LINEAR2)
# auc=0.9779   1: get_clf13 ClfSpacy(dropout=0.5, learn_rate=0.002, lstm_type=8, max_length=100, n_hidden=128, predict_method=MEAN)

#  spacy_lstm12.log instance3 has
# RESULTS SUMMARY: 61
# auc=0.9896  34: get_clf12 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=6, max_length=100, n_hidden=256, predict_method=LINEAR2)
# auc=0.9891  60: get_clf12 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=2, max_length=100, n_hidden=256, predict_method=LINEAR2)
# auc=0.9885  55: get_clf12 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=2, max_length=100, n_hidden=256, predict_method=MEAN_MAX)
# auc=0.9873  33: get_clf12 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=6, max_length=100, n_hidden=256, predict_method=LINEAR)


#  spacy_lstm11.log  has
# RESULTS SUMMARY: 59
# auc=0.9837  25: get_clf1 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=6, max_length=100, n_hidden=128, predict_method=LINEAR2)
# auc=0.9833  14: get_clf1 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=7, max_length=100, n_hidden=128, predict_method=LINEAR2)
# auc=0.9831  21: get_clf1 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=7, max_length=100, n_hidden=128, predict_method=PC90)


# clf=0.9837   6: get_clf1 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=2, max_length=100, n_examples=-1, n_hidden=128)
# clf=0.9815   2: get_clf0 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=2, max_length=100, n_examples=-1, n_hidden=64)
# clf=0.9804   3: get_clf0 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=1, max_length=100, n_examples=-1, n_hidden=64)
# clf=0.9741   4: get_clf1 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=4, max_length=100, n_examples=-1, n_hidden=128)
# clf=0.9733   1: get_clf0 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=3, max_length=100, n_examples=-1, n_hidden=64)
# clf=0.9721   0: get_clf0 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=4, max_length=100, n_examples=-1, n_hidden=64)
# clf=0.9708   5: get_clf1 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=3, max_length=100, n_examples=-1, n_hidden=128)

# RESULTS SUMMARY: 4
# auc=0.9760   0: get_clf1 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=2, max_length=100, n_examples=-1, n_hidden=128)
# auc=0.9730   1: get_clf0 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=2, max_length=100, n_examples=-1, n_hidden=64)
# auc=0.9725   2: get_clf2 ClfSpacy(dropout=0.5, learn_rate=0.001, lstm_type=2, max_length=150, n_examples=-1, n_hidden=64)
# auc=0.9670   3: get_clf3 ClfSpacy(dropout=0.2, learn_rate=0.001, lstm_type=2, max_length=100, n_examples=-1, n_hidden=64)

xprint_init(submission_name, False)
clf_list = [get_clf12, get_clf13, get_clf14, get_clf15,
           # get_clf0, get_clf2, get_clf3,
           get_clf4, get_clf5,
           get_clf1]
# clf_list.reverse()
auc_list = []
frozen = True
completed_tests = load_json(run_summary_path, {})
n_completed0 = len(completed_tests)
n_runs0 = 0  # min(len(v) for v in (list(completed_tests.values()) + [[]]))
print('n_completed0=%d n_runs0=%d' % (n_completed0, n_runs0))

for get_clf in clf_list:
    for lstm_type in (8, 7, 6, 5, 2):  # , 3, 4, 1):
        for frozen in [True]:  # (False, True, False):
            for predict_method in PREDICT_METHODS:
                xprint('#' * 80)
                clf_str = str(get_clf())
                xprint(clf_str)
                runs = completed_tests.get(clf_str, [])
                if len(runs) > n_runs0:
                    xprint('skipping')
                    continue

                set_random_seed(1234)
                evaluator = Evaluator(n=1)
                ok, auc = evaluator.evaluate(get_clf)

                auc_list.append((auc, get_clf.__name__, str(get_clf())))
                show_results(auc_list)

                runs.append(auc_score_list(auc))
                completed_tests[clf_str] = runs
                save_json(run_summary_path, completed_tests)
                xprint('n_completed=%d = %d + %d' % (len(completed_tests), n_completed0,
                    len(completed_tests) - n_completed0))
                xprint('&' * 100)

xprint('$' * 100)
