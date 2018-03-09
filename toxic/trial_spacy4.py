# coding: utf-8
"""
    SpaCy deep_learning_keras.py solution to Kaggle Toxic Comment challenge
"""
import os
from utils import xprint_init, xprint, load_json, save_json
from framework import (SUMMARY_DIR, Evaluator, set_random_seed, show_auc, set_n_samples,
    get_n_samples_str)
from clf_spacy import ClfSpacy, PREDICT_METHODS


submission_name = 'spacy_lstm16'
epochs = 40
set_n_samples(-1)
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


clf_list = [get_clf12, get_clf13]
lstm_list = [6, 8]
frozen_list = [False]
frozen = True

xprint_init(submission_name, False)
auc_list = []
run_summary = load_json(run_summary_path, {'completed': []})
completed_tests = set(run_summary.get('completed', []))

for get_clf in clf_list:
    for lstm_type in lstm_list:
        for predict_method in PREDICT_METHODS:
            xprint('#' * 80)
            clf_str = str(get_clf())
            xprint(clf_str)
            if clf_str in completed_tests:
                xprint('skipping')
                continue
            set_random_seed(1234)
            evaluator = Evaluator(n=1)
            ok, auc0 = evaluator.evaluate(get_clf)
            auc_list.append((auc0, get_clf.__name__, str(get_clf())))
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

            completed_tests.add(clf_str)
            run_summary['completed'] = sorted(completed_tests)
            save_json(run_summary_path, run_summary)
            xprint('n_completed=%d' % len(completed_tests))
            xprint('&' * 100)

xprint('$' * 100)
