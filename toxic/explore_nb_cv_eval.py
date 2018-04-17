import random
import time
from utils import xprint, xprint_init
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from functools import partial
from gru_framework import (set_n_samples, get_n_samples_str, prepare_data, CV_predictor,
    show_results_cv, save_submission)
from clf_tfidf_nb import ClfTfidfNB


def get_clf_base(do_xgb):
    def get_est():
        if do_xgb:
            assert False
            return XGBClassifier()
        else:
            return LogisticRegression(C=4, dual=True)

    return ClfTfidfNB(get_est, do_spacy=False)


# set_n_samples(60000)
submission_base = 'tfidf_nb_eval.%s' % get_n_samples_str()
xprint_init(submission_base, False)


params_list = []
for do_xgb in [False]:
    params = [do_xgb]
    params_list.append(params)

xprint('params_list=%d' % len(params_list))
# params0, params_list1 = params_list[0], params_list[1:]
# random.seed(time.time())
# random.shuffle(params_list1)
# assert len(params0) == len(params_list1[0])
# params_list = [params0] + params_list1

xprint('params_list=%d' % len(params_list))
for i, params in enumerate(params_list[:10]):
    print(i, params)
xprint('$' * 100)

(idx_train, X_train, y_train), (idx_test, X_test, y_test) = prepare_data(0.2)
n_splits = 4

auc_list = []
for params in params_list:
    do_xgb, = params
    assert not do_xgb
    n = len(auc_list)
    get_clf = partial(get_clf_base, do_xgb)

    xprint('#' * 100)
    xprint('config=%d of %d' % (n, len(params_list) * 4))
    xprint('params=%s %s' % (get_clf_base.__name__, list(params)))
    evaluator = CV_predictor(get_clf, idx_train, X_train, y_train, idx_test, X_test, y_test,
        n_splits)
    evaluator.predict()
    auc_list.append((evaluator.auc_train, str(get_clf())))
    show_results_cv(auc_list)
    evaluator.eval_predictions()
    # submission_name = '%s.%03d.csv' % (submission_base, n)
    # save_submission(evaluator.test_predictions, submission_name)
    xprint('&' * 100)

xprint('$' * 100)
