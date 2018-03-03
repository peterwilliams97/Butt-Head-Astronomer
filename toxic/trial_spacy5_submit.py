# coding: utf-8
"""
    SpaCy deep_learning_keras.py solution to Kaggle Toxic Comment challenge
"""
from utils import xprint_init, xprint
from framework import Evaluator, set_random_seed, make_submission_reductions, set_n_samples
from clf_spacy import ClfSpacy, PREDICT_METHODS_GOOD


submission_name = 'spacy_lstm20'
do_submission = False
set_n_samples(30000)
epochs = 8


def get_clf():
    return ClfSpacy(n_hidden=512, max_length=100,  # Shape
                    dropout=0.5, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=150, frozen=True,
                    lstm_type=6, predict_method=PREDICT_METHODS_GOOD[0])


xprint_init(submission_name, do_submission)
xprint('#' * 80)
xprint(get_clf())
set_random_seed(seed=1234)

if do_submission:
    make_submission_reductions(get_clf, submission_name, PREDICT_METHODS_GOOD)
else:
    evaluator = Evaluator(n=1)
    ok, auc = evaluator.evaluate_reductions(get_clf, PREDICT_METHODS_GOOD)
xprint('$' * 80)
