# coding: utf-8
"""
    SpaCy deep_learning_keras.py solution to Kaggle Toxic Comment challenge
"""
from utils import xprint_init, xprint, touch
from framework import Evaluator, set_random_seed, make_submission_reductions, get_n_samples_str
from clf_spacy import ClfSpacy, PREDICT_METHODS_GOOD


submission_name = 'spacy_lstmx_90'
do_submission = False
epochs = 9
if not do_submission:
    epochs = 40


# gpu3: spacy_lstm21_flip.40000.log
# auc=0.9840   9: get_clf25 ClfSpacy(batch_size=300, dropout=0.5, learn_rate=0.001, lstm_type=9, max_length=75, n_hidden=512, pm=LINEAR)
# auc=0.9840  10: get_clf25 ClfSpacy(batch_size=300, dropout=0.5, learn_rate=0.001, lstm_type=9, max_length=75, n_hidden=512, pm=LINEAR2)
# auc=0.9839  11: get_clf25 ClfSpacy(batch_size=300, dropout=0.5, learn_rate=0.001, lstm_type=9, max_length=75, n_hidden=512, pm=LINEAR3)


def get_clf():
    return ClfSpacy(n_hidden=512, max_length=75,  # Shape
                    dropout=0.5, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=300, frozen=True,
                    lstm_type=9, predict_method=PREDICT_METHODS_GOOD[0])


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
touch('completed.spacy_lstmx_90.txt')

"""
instance5/spacy_lstm20s.ALL.LINEAR2.csv
Your submission scored 0.9723, which is not an improvement of your best score. Keep trying!

"""
