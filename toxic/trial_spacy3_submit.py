# coding: utf-8
"""
    SpaCy deep_learning_keras.py solution to Kaggle Toxic Comment challenge
"""
from utils import xprint_init, xprint
from framework import Evaluator, set_random_seed, make_submission, set_n_samples
from clf_spacy import ClfSpacy


submission_name = 'spacy_lstm10'
do_submission = True
epochs = 6


def get_clf():
    return ClfSpacy(n_hidden=128, max_length=100,  # Shape
                    dropout=0.5, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=150, frozen=True,
                    lstm_type=2)


xprint_init(submission_name, do_submission)
xprint('#' * 80)
xprint(get_clf())
set_random_seed(seed=1234)

if do_submission:
    make_submission(get_clf, submission_name)
else:
    evaluator = Evaluator(n=3)
    ok, auc = evaluator.evaluate(get_clf)
xprint('$' * 80)
