# coding: utf-8
"""
    SpaCy deep_learning_keras.py solution to Kaggle Toxic Comment challenge
"""
import os
import time
import random
from utils import xprint_init, xprint, load_json, save_json, touch
from framework import (SUMMARY_DIR, Evaluator, set_random_seed, set_n_samples, get_n_samples_str,
    auc_score_list, show_results)
from clf_vector import ClfVector
from reductions import PREDICT_METHODS_GOOD


submission_name = 'v_trial_fasttest_005'
epochs = 40
random_seed = 60018
set_n_samples(40000)
run_summary_path = os.path.join(SUMMARY_DIR,
    '%s.%s.run_summary.json' % (submission_name, get_n_samples_str()))


# ClfVector(batch_size=150, dropout=0.1, embed_name=6B, embed_size=50, epochs=2,
#     frozen=False, learn_rate=0.001, learn_rate_unfrozen=0.0, lowercase=True, lstm_type=6,
#      max_features=20000, max_length=10, n_hidden=16, predict_method=MEAN)

def get_clf():
    return ClfVector(max_features=max_features, max_length=max_length,  # Shape
                     n_hidden=n_hidden,
                     dropout=dropout,
                     learn_rate=learn_rate,
                     learn_rate_unfrozen=learn_rate * lr_ratio,
                     batch_size=batch_size,
                     epochs=40, lstm_type=lstm_type, predict_method=predict_method,
                     epochs2=40, randomized=randomized,
                     token_method=4)


params_list = []
for lstm_type in [13]:
    for max_length in [50, 75, 100]:  # [50, 75, 100, 150]:
        for max_features in [50000]:  # [20000, 25000, 30000, 40000]:
            for n_hidden in [120, 150]:
                for dropout in [0.2, 0.4]:  # [0.1, 0.3, 0.5]:
                        for batch_size in [300]:
                            params = (lstm_type, max_length, max_features, n_hidden, dropout)
                            params_list.append(params)

print('params_list=%d' % len(params_list))
random.seed(time.time())
random.shuffle(params_list)
# params_list.reverse()
print('params_list=%d' % len(params_list))
for i, params in enumerate(params_list[:10]):
    print(i, params)
print('$' * 100)

xprint_init(submission_name, False)
auc_list = []
completed_tests = load_json(run_summary_path, {})
xprint('run_summary_path=%s' % run_summary_path)
n_completed0 = len(completed_tests)

for n_runs0 in range(2):
    print('n_completed0=%d n_runs0=%d' % (n_completed0, n_runs0))

    for p_i, (lstm_type, max_length, max_features, n_hidden, dropout) in enumerate(params_list):
            for learn_rate in [0.0005, 0.001, 0.002, 0.004]:
                for lr_ratio in [2.0, 1.0, 0.5]:
                    for randomized in [True, False]:

                        xprint('#' * 80)
                        predict_method = PREDICT_METHODS_GOOD[0]
                        clf_str = str(get_clf())
                        xprint('params %d: %s' % (p_i, clf_str))
                        runs = completed_tests.get(clf_str, [])
                        if len(runs) > n_runs0:
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
