# coding: utf-8
"""
    SpaCy deep_learning_keras.py solution to Kaggle Toxic Comment challenge
"""
from utils import xprint_init, xprint
from framework import Evaluator, seed_random, show_auc
from clf_spacy import ClfSpacy


do_submission = False
submission_name = 'spacy_lstm4'
epochs = 40
frozen = True
submission_name = '%s.%s' % (submission_name, frozen)


def get_clf0():
    return ClfSpacy(n_hidden=64, max_length=100,  # Shape
                    dropout=0.5, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=150, n_examples=-1, frozen=frozen)


def get_clf1():
    return ClfSpacy(n_hidden=128, max_length=100,  # Shape
                    dropout=0.5, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=150, n_examples=-1, frozen=frozen)


def get_clf2():
    return ClfSpacy(n_hidden=64, max_length=150,  # Shape
                    dropout=0.5, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=150, n_examples=-1, frozen=frozen)


def get_clf3():
    return ClfSpacy(n_hidden=64, max_length=100,  # Shape
                    dropout=0.2, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=150, n_examples=-1, frozen=frozen)


def get_clf4():
    return ClfSpacy(n_hidden=64, max_length=100,  # Shape
                    dropout=0.5, learn_rate=0.002,  # General NN config
                    epochs=epochs, batch_size=150, n_examples=-1, frozen=frozen)


def get_clf5():
    return ClfSpacy(n_hidden=64, max_length=100,  # Shape
                    dropout=0.5, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=50, n_examples=-1, frozen=frozen)


def get_clf6():
    return ClfSpacy(n_hidden=64, max_length=50,  # Shape
                    dropout=0.5, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=50, n_examples=-1, frozen=frozen)


xprint_init(submission_name, do_submission)
clf_list = [get_clf0, get_clf1, get_clf2, get_clf3, get_clf4, get_clf5]
auc_list = []
for get_clf in clf_list:
    xprint('#' * 80)
    xprint(get_clf())
    seed_random(seed=1234)
    evaluator = Evaluator(n=1)
    ok, auc = evaluator.evaluate(get_clf)
    auc_list.append(auc)
    xprint('~' * 80)
    for i, auc in enumerate(auc_list):
        clf = clf_list[i]
        xprint('clf=%d %s frozen=%s' % (i, clf, frozen))
        xprint(clf())
        show_auc(auc)

xprint('$' * 80)
