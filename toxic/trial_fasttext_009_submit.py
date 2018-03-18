# coding: utf-8
"""
    SpaCy deep_learning_keras.py solution to Kaggle Toxic Comment challenge
"""
import os
from utils import xprint_init, xprint, load_json, save_json, touch
from framework import (SUMMARY_DIR, Evaluator, set_random_seed, set_n_samples, get_n_samples_str,
    auc_score_list, show_results, make_submission_reductions)
from clf_vector import ClfVector
from reductions import PREDICT_METHODS_GOOD


# auc=0.9881   5: get_clf ClfVector(batch_size=300, do_spacy=False, dropout=0.4, epochs=40, epochs2=40, learn_rate=0.0005, learn_rate_unfrozen=0.001, lstm_type=13, max_features=50000, max_length=75, n_hidden=120, predict_method=BEST, randomized=False, single_oov=False, token_method=4)
# auc=0.9863   5: get_clf ClfVector(batch_size=300, do_spacy=False, dropout=0.2, epochs=40, epochs2=0, learn_rate=0.004, learn_rate_unfrozen=0.002, lstm_type=13, max_features=50000, max_length=50, n_hidden=80, predict_method=BEST, randomized=False, single_oov=False, token_method=4)

submission_name = 'v_trial_fasttest_009_full_all'
epochs2 = 40
random_seed = 60018
do_submission = True
# set_n_samples(20000)
run_summary_path = os.path.join(SUMMARY_DIR,
    '%s.%s.run_summary.json' % (submission_name, get_n_samples_str()))


# ClfVector(batch_size=150, dropout=0.1, embed_name=6B, embed_size=50, epochs=2,
#     frozen=False, learn_rate=0.001, learn_rate_unfrozen=0.0, lowercase=True, lstm_type=6,
#      max_features=20000, max_length=10, n_hidden=16, predict_method=MEAN)

def get_clf():
    return ClfVector(batch_size=300, do_spacy=False,
                     dropout=0.4, epochs=1, epochs2=1,
                     learn_rate=0.0005, learn_rate_unfrozen=0.001,
                     lstm_type=13,
                     max_features=50000, max_length=75,
                     n_hidden=120, predict_method=predict_method,
                     randomized=False, single_oov=False, token_method=4)


print('$' * 100)

xprint_init(submission_name, do_submission)

if do_submission:
    predict_method = PREDICT_METHODS_GOOD[0]
    make_submission_reductions(get_clf, submission_name, PREDICT_METHODS_GOOD)
else:

    auc_list = []
    completed_tests = load_json(run_summary_path, {})
    xprint('run_summary_path=%s' % run_summary_path)
    n_completed0 = len(completed_tests)

    for n_runs0 in range(4):
        print('n_completed0=%d n_runs0=%d' % (n_completed0, n_runs0))

        xprint('#' * 80)
        predict_method = PREDICT_METHODS_GOOD[0]
        clf_str = str(get_clf())
        xprint(clf_str)

        runs = completed_tests.get(clf_str, [])
        if len(runs) > n_runs0 * (len(PREDICT_METHODS_GOOD) + 1):
            xprint('skipping runs=%d n_runs0=%d' % (len(runs), n_runs0))
            continue

        set_random_seed(random_seed + n_runs0)
        evaluator = Evaluator(n=1)
        ok, auc_reductions, best_method = evaluator.evaluate_reductions(get_clf,
            PREDICT_METHODS_GOOD)
        assert ok

        for predict_method in sorted(auc_reductions):
            auc = auc_reductions[predict_method]
            xprint('<->.' * 25)
            xprint('predict_method=%s' % predict_method)
            if predict_method == 'BEST':
                xprint('best_method=%s' % best_method)
            assert auc.all() > 0.0, auc

            auc_list.append((auc, get_clf.__name__, str(get_clf())))
            show_results(auc_list)

            runs.append(auc_score_list(auc))
            completed_tests[str(get_clf())] = runs
            save_json(run_summary_path, completed_tests)
            xprint('n_completed=%d = %d + %d' % (len(completed_tests), n_completed0,
                len(completed_tests) - n_completed0))
            xprint('&' * 100)

touch('completed.v_trial_vector_007.txt')
xprint('$' * 100)
