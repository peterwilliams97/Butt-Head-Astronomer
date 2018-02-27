# coding: utf-8
"""
    SpaCy deep_learning_keras.py solution to Kaggle Toxic Comment challenge
"""
from utils import xprint_init, xprint
from framework import Evaluator, seed_random, show_auc
from clf_spacy import ClfSpacy


submission_name = 'spacy_lstm8'
epochs = 40


def get_clf0():
    return ClfSpacy(n_hidden=64, max_length=100,  # Shape
                    dropout=0.5, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=150, frozen=frozen,
                    lstm_type=lstm_type)


def get_clf1():
    return ClfSpacy(n_hidden=128, max_length=100,  # Shape
                    dropout=0.5, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=150, frozen=frozen,
                    lstm_type=lstm_type)


def get_clf2():
    return ClfSpacy(n_hidden=64, max_length=150,  # Shape
                    dropout=0.5, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=150, frozen=frozen,
                    lstm_type=lstm_type)


def get_clf3():
    return ClfSpacy(n_hidden=64, max_length=100,  # Shape
                    dropout=0.2, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=150, frozen=frozen,
                    lstm_type=lstm_type)


def get_clf4():
    return ClfSpacy(n_hidden=64, max_length=100,  # Shape
                    dropout=0.5, learn_rate=0.002,  # General NN config
                    epochs=epochs, batch_size=150, frozen=frozen,
                    lstm_type=lstm_type)


def get_clf5():
    return ClfSpacy(n_hidden=64, max_length=100,  # Shape
                    dropout=0.5, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=50, frozen=frozen,
                    lstm_type=lstm_type)


def get_clf6():
    return ClfSpacy(n_hidden=64, max_length=50,  # Shape
                    dropout=0.5, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=50, frozen=frozen,
                    lstm_type=lstm_type)


xprint_init(submission_name, False)
clf_list = [get_clf1, get_clf0, get_clf2, get_clf3, get_clf4, get_clf5]
# clf_list.reverse()
auc_list = []
frozen = True

# clf=0.9837   6: get_clf1 ClfSpacy(batch_size=150, dropout=0.5, learn_rate=0.001, lstm_type=2, max_length=100, n_examples=-1, n_hidden=128)
# clf=0.9815   2: get_clf0 ClfSpacy(batch_size=150, dropout=0.5, learn_rate=0.001, lstm_type=2, max_length=100, n_examples=-1, n_hidden=64)
# clf=0.9804   3: get_clf0 ClfSpacy(batch_size=150, dropout=0.5, learn_rate=0.001, lstm_type=1, max_length=100, n_examples=-1, n_hidden=64)
# clf=0.9741   4: get_clf1 ClfSpacy(batch_size=150, dropout=0.5, learn_rate=0.001, lstm_type=4, max_length=100, n_examples=-1, n_hidden=128)
# clf=0.9733   1: get_clf0 ClfSpacy(batch_size=150, dropout=0.5, learn_rate=0.001, lstm_type=3, max_length=100, n_examples=-1, n_hidden=64)
# clf=0.9721   0: get_clf0 ClfSpacy(batch_size=150, dropout=0.5, learn_rate=0.001, lstm_type=4, max_length=100, n_examples=-1, n_hidden=64)
# clf=0.9708   5: get_clf1 ClfSpacy(batch_size=150, dropout=0.5, learn_rate=0.001, lstm_type=3, max_length=100, n_examples=-1, n_hidden=128)

for lstm_type in (2, 8, 7, 6, 5):  # , 3, 4, 1):
    for get_clf in clf_list:
        xprint('#' * 80)
        xprint(get_clf())
        seed_random(seed=1234)
        evaluator = Evaluator(n=2)
        ok, auc = evaluator.evaluate(get_clf)
        auc_list.append((auc, get_clf.__name__, str(get_clf())))
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
        xprint('&' * 100)


xprint('$' * 100)
