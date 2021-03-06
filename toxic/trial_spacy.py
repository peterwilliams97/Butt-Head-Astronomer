# coding: utf-8
"""
    SpaCy deep_learning_keras.py solution to Kaggle Toxic Comment challenge
"""
from utils import xprint_init, xprint
from framework import Evaluator, seed_random, make_submission
from clf_spacy import ClfSpacy


do_submission = True
submission_name = 'spacy_lstm2'


def get_clf():
    return ClfSpacy(n_hidden=64, max_length=100,  # Shape
                    dropout=0.5, learn_rate=0.001,  # General NN config
                    epochs=3, batch_size=150)


xprint_init(submission_name, do_submission)
xprint('#' * 80)
xprint(get_clf())
seed_random(seed=1234)

if do_submission:
    make_submission(get_clf, submission_name)
else:
    evaluator = Evaluator(n=1)
    ok, auc = evaluator.evaluate(get_clf)
xprint('$' * 80)
