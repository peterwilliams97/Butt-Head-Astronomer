# coding: utf-8
"""
    SpaCy deep_learning_keras.py solution to Kaggle Toxic Comment challenge
"""
from utils import xprint_init, xprint, touch
from framework import Evaluator, set_random_seed, make_submission_reductions, get_n_samples_str
from clf_spacy import ClfSpacy, PREDICT_METHODS_GOOD


submission_name = 'spacy_lstmx_110'
do_submission = True
epochs = 6
if not do_submission:
    epochs = 40

# gpu2: auc=0.982  best_epoch=6
# ClfSpacy(batch_size=300, dropout=0.3, epochs=6, epochs2=2, frozen=True, learn_rate=0.001, lstm_type=6, max_length=100, n_hidden=512, predict_method=MEAN)


def get_clf():
    return ClfSpacy(n_hidden=512, max_length=100,  # Shape
                    dropout=0.3, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=300, frozen=True,
                    lstm_type=6, predict_method=PREDICT_METHODS_GOOD[0])


xprint_init('%s.%s' % (submission_name, get_n_samples_str()), do_submission)

xprint('#' * 80)
xprint(get_clf())
set_random_seed(seed=1234)

if do_submission:
    make_submission_reductions(get_clf, submission_name, PREDICT_METHODS_GOOD)
else:
    evaluator = Evaluator(n=1)
    ok, auc = evaluator.evaluate_reductions(get_clf, PREDICT_METHODS_GOOD)
xprint('$' * 80)
touch('completed.spacy_lstmx_110.txt')

"""
instance5/spacy_lstm20s.ALL.LINEAR2.csv
Your submission scored 0.9723, which is not an improvement of your best score. Keep trying!

"""
