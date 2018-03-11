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
from clf_pipe import ClfPipe
from reductions import PREDICT_METHODS_GOOD
from embeddings import GLOVE_COMBOS


submission_name = 'p_trial_pipe_006'
epochs = 40
random_seed = 50006
set_n_samples(19999)
run_summary_path = os.path.join(SUMMARY_DIR,
    '%s.%s.run_summary.json' % (submission_name, get_n_samples_str()))


def get_clf23():
    return ClfPipe(embed_name=embed_name, embed_size=embed_size,
                   n_hidden=256, max_length=100,  # Shape
                   dropout=0.3, learn_rate=0.001,  # General NN config
                   epochs=epochs, batch_size=150,
                   lstm_type=lstm_type, predict_method=PREDICT_METHODS_GOOD[0])


get_clf = get_clf23
params_list = []
for lstm_type in [6, 9]:
    for embed_name, embed_size in GLOVE_COMBOS:
        params = (lstm_type, embed_name, embed_size)
        params_list.append(params)

print('params_list=%d' % len(params_list))
random.seed(time.time())
random.shuffle(params_list)
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
    for p_i, (lstm_type, embed_name, embed_size) in enumerate(params_list):
            xprint('#' * 80)
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

touch('completed.p_trial_pipe_006.txt')
xprint('$' * 100)
