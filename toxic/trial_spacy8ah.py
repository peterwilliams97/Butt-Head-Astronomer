# coding: utf-8
"""
    SpaCy deep_learning_keras.py solution to Kaggle Toxic Comment challenge
"""
import os
from utils import xprint_init, xprint, load_json, save_json, touch
from framework import (SUMMARY_DIR, Evaluator, set_random_seed, set_n_samples, get_n_samples_str,
    auc_score_list, show_results)
from clf_spacy import ClfSpacy, PREDICT_METHODS_GOOD


submission_name = 'e_spacy_lstmx_83a'
epochs = 40
frozen = False
set_n_samples(20000)
random_seed = 2238
run_summary_path = os.path.join(SUMMARY_DIR,
    '%s.%s.run_summary.json' % (submission_name, get_n_samples_str()))


def get_clf31():
    return ClfSpacy(n_hidden=n_hidden, max_length=75,  # Shape
                    dropout=0.3, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=150, frozen=frozen,
                    lstm_type=lstm_type, predict_method=predict_method)


def get_clf32():
    return ClfSpacy(n_hidden=n_hidden, max_length=75,  # Shape
                    dropout=0.3, learn_rate=0.001, learn_rate_unfrozen=0.0,  # General NN config
                    epochs=epochs, batch_size=150, frozen=frozen,
                    lstm_type=lstm_type, predict_method=predict_method)


def get_clf33():
    return ClfSpacy(n_hidden=n_hidden, max_length=75, max_features=30000,  # Shape
                    dropout=0.3, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=150, frozen=frozen,
                    lstm_type=lstm_type, predict_method=predict_method)


def get_clf34():
    return ClfSpacy(n_hidden=n_hidden, max_length=75, max_features=15000,  # Shape
                    dropout=0.3, learn_rate=0.005,  # General NN config
                    epochs=epochs, batch_size=150, frozen=frozen,
                    lstm_type=lstm_type, predict_method=predict_method)


def get_clf35():
    return ClfSpacy(n_hidden=n_hidden, max_length=100,  # Shape
                    dropout=0.3, learn_rate=0.005,   # General NN config
                    epochs=epochs, batch_size=150, frozen=frozen,
                    lstm_type=lstm_type, predict_method=predict_method)


def get_clf36():
    return ClfSpacy(n_hidden=n_hidden, max_length=100,  # Shape
                    dropout=0.3, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=50, frozen=frozen,
                    lstm_type=lstm_type, predict_method=predict_method)


def get_clf37():
    return ClfSpacy(n_hidden=n_hidden, max_length=75,  # Shape
                    dropout=0.3, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=150, frozen=frozen,
                    lstm_type=lstm_type, predict_method=predict_method)


clf_list = [get_clf31, get_clf32, get_clf33, get_clf34, get_clf35, get_clf36, get_clf37]
lstm_list = [6, 9, 7, 8]
n_hidden_list = [64, 128, 256]

xprint_init('%s.%s' % (submission_name, get_n_samples_str()), False)
auc_list = []
completed_tests = load_json(run_summary_path, {})
xprint('run_summary_path=%s' % run_summary_path),
n_completed0 = len(completed_tests)

for n_runs0 in range(3):
    print('n_completed0=%d n_runs0=%d' % (n_completed0, n_runs0))

    for n_hidden in n_hidden_list:
        for lstm_type in lstm_list:
            for get_clf in clf_list:

                xprint('#' * 80)
                predict_method = PREDICT_METHODS_GOOD[0]
                clf_str = str(get_clf())
                xprint(clf_str)
                runs = completed_tests.get(clf_str, [])
                if len(runs) > n_runs0:
                    xprint('skipping runs=%d n_runs0=%d' % (len(runs), n_runs0))
                    continue

                set_random_seed(random_seed + n_runs0)
                evaluator = Evaluator(n=1)
                ok, auc_reductions, best_method = evaluator.evaluate_reductions(get_clf,
                    PREDICT_METHODS_GOOD)
                assert ok

                for predict_method in sorted(auc_reductions):
                    auc = auc_reductions[predict_method]
                    xprint('<->.' * 25)
                    xprint('predict_method=%s' % predict_method)
                    if predict_method == 'BEST':
                        xprint('best_method=%s' % best_method)
                    assert auc.all() > 0.0, auc

                    auc_list.append((auc, get_clf.__name__, str(get_clf())))
                    show_results(auc_list)

                    runs.append(auc_score_list(auc))
                    completed_tests[str(get_clf())] = runs
                    save_json(run_summary_path, completed_tests)
                    xprint('n_completed=%d = %d + %d' % (len(completed_tests), n_completed0,
                        len(completed_tests) - n_completed0))
                xprint('&' * 100)

touch('completed.e_spacy_lstmx_83a.txt')
xprint('$' * 100)
