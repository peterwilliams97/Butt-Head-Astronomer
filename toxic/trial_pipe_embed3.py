# coding: utf-8
"""
    SpaCy deep_learning_keras.py solution to Kaggle Toxic Comment challenge
"""
import os
from utils import xprint_init, xprint, load_json, save_json
from framework import (SUMMARY_DIR, Evaluator, set_random_seed, set_n_samples, get_n_samples_str,
    auc_score_list, show_results)
from clf_pipe import ClfPipe
from reductions import PREDICT_METHODS_GOOD
from embeddings import GLOVE_COMBOS


submission_name = 'trial_pipe_003'
epochs = 40
random_seed = 50003
set_n_samples(19999)
run_summary_path = os.path.join(SUMMARY_DIR,
    '%s.%s.run_summary.json' % (submission_name, get_n_samples_str()))


# ClfPipe(batch_size=150, dropout=0.1, embed_name=6B, embed_size=50, epochs=2,
#     frozen=False, learn_rate=0.001, learn_rate_unfrozen=0.0, lowercase=True, lstm_type=6,
#      max_features=20000, max_length=10, n_hidden=16, predict_method=MEAN)

def get_clf42():
    print('$$$', embed_name, embed_size)
    return ClfPipe(embed_name=embed_name, embed_size=embed_size,
                   max_features=20000, max_length=100,  # Shape
                   n_hidden=64,
                   dropout=0.5, learn_rate=0.001,  # General NN config
                   batch_size=150,
                   epochs=epochs, lstm_type=lstm_type, predict_method=predict_method)


clf_list = [get_clf42]  # , get_clf22, get_clf23, get_clf24, get_clf25]
lstm_list = [6, 9]

xprint_init(submission_name, False)
auc_list = []
completed_tests = load_json(run_summary_path, {})
xprint('run_summary_path=%s' % run_summary_path)
n_completed0 = len(completed_tests)

for n_runs0 in range(2):
    print('n_completed0=%d n_runs0=%d' % (n_completed0, n_runs0))
    for get_clf in clf_list:
        for lstm_type in lstm_list:
            for embed_name, embed_size in GLOVE_COMBOS:
                for predict_method in PREDICT_METHODS_GOOD:
                    xprint('#' * 80)
                    predict_method = PREDICT_METHODS_GOOD[0]
                    clf_str = str(get_clf())
                    xprint(clf_str)
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

touch('completed.trial_pipe_003.txt')
xprint('$' * 100)
