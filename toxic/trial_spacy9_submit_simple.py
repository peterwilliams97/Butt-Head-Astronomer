# coding: utf-8
"""
    SpaCy deep_learning_keras.py solution to Kaggle Toxic Comment challenge
"""
from utils import xprint_init, xprint, touch
from framework import Evaluator, set_random_seed, make_submission, get_n_samples_str, set_n_samples
from clf_spacy import ClfSpacy, PREDICT_METHODS_GOOD

do_submission = False
set_n_samples(39999)
set_random_seed(5001)
submission_name = 'spacy_lstmx_901'
epochs = 9
if not do_submission:
    epochs = 40


# gpu3: spacy_lstm21_flip.40000.log
# auc=0.9840   9: get_clf25 ClfSpacy(batch_size=300, dropout=0.5, learn_rate=0.001, lstm_type=9, max_length=75, n_hidden=512, pm=LINEAR)
# auc=0.9840  10: get_clf25 ClfSpacy(batch_size=300, dropout=0.5, learn_rate=0.001, lstm_type=9, max_length=75, n_hidden=512, pm=LINEAR2)
# auc=0.9839  11: get_clf25 ClfSpacy(batch_size=300, dropout=0.5, learn_rate=0.001, lstm_type=9, max_length=75, n_hidden=512, pm=LINEAR3)


def get_clf():
    return ClfSpacy(n_hidden=64, max_length=75, max_features=20000, # Shape
                    dropout=0.5, learn_rate=0.001, frozen=False, # General NN config
                    epochs=epochs, batch_size=300,
                    lstm_type=6, predict_method=PREDICT_METHODS_GOOD[0])


xprint_init('%s.%s' % (submission_name, get_n_samples_str()), do_submission)

xprint('#' * 80)
xprint(get_clf())
set_random_seed(seed=1234)

if do_submission:
    make_submission(get_clf, submission_name)
else:
    evaluator = Evaluator(n=1)
    ok, auc = evaluator.evaluate(get_clf)
xprint('$' * 80)
touch('completed.spacy_lstmx_90.txt')

"""
ClfSpacy(n_hidden=64, max_length=75, max_features=20000, # Shape
                    dropout=0.5, learn_rate=0.001,  # General NN config
                    epochs=epochs, batch_size=300,
                    lstm_type=6, predict_method=PREDICT_METHODS_GOOD[0])
 Mean: auc=0.975 (toxic:0.965, severe_toxic:0.982, obscene:0.979, threat:0.989, insult:0.976, identity_hate:0.961)

"""
