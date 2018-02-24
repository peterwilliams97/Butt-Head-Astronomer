# coding: utf-8
"""
    SpaCy deep_learning_keras.py solution to Kaggle Toxic Comment challenge
"""
from utils import xprint_init, xprint
from framework import Evaluator, seed_random
from clf_spacy import ClfSpacy


submission_name = 'spacy_lstm1'


def get_clf():
    return ClfSpacy(n_hidden=64, max_length=100,  # Shape
                    dropout=0.5, learn_rate=0.001,  # General NN config
                    epochs=20, batch_size=150, n_examples=-1)


xprint('#' * 80)
xprint_init(submission_name)
xprint(get_clf())
seed_random(seed=1234)

evaluator = Evaluator(n=1)
ok, auc = evaluator.evaluate(get_clf)
xprint('$' * 80)
