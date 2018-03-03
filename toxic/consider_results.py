# coding: utf-8
"""
    SpaCy deep_learning_keras.py solution to Kaggle Toxic Comment challenge
"""
import os
from glob import glob
import numpy as np
from collections import defaultdict
from utils import xprint_init, xprint, load_json
from framework import LABEL_COLS


np.set_printoptions(precision=6)


def simplify(clf_str):
    return clf_str.replace('epochs=40, epochs2=2, frozen=True, ', ''
        ).replace('batch_size', 'batch'
        ).replace('learn_rate', 'lr'
        ).replace('predict_method', 'm')


def display_results(completed_tests, do_max):
    n_completed = len(completed_tests)
    n_runs = min(len(v) for v in completed_tests.values())
    auc = np.zeros((n_completed, n_runs), dtype=np.float64)

    l1 = set(completed_tests)
    l2 = {(clf_str) for clf_str in l1}
    assert len(l1) == len(l2), sorted(l1 - l2)

    clf_auc = {}
    for clf_str, runs in completed_tests.items():
        clf_str = simplify(clf_str)
        n_runs = min(len(v) for v in runs)
        runs = runs[:n_runs]
        # print('runs=%d %s' % (len(runs), clf_str))
        auc = np.zeros((n_runs, len(LABEL_COLS)), dtype=np.float64)
        for i, v in enumerate(runs):
            # print('v=%s' % v)
            auc[i, :] = np.array(v[1], dtype=np.float64)
            reduced_auc = auc.max(axis=0) if do_max else auc.mean(axis=0)
        duplicate = False
        if clf_auc:
            previous_auc = [v for _, v in clf_auc.values()]
            for p in previous_auc:
                d = reduced_auc - p
                if not np.abs(d).any() > 1e-6:
                    duplicate = True
                    break
                assert np.abs(d).any() > 1e-6, (reduced_auc, p, d)
        if duplicate:
            continue
        clf_auc[clf_str] = (n_runs, reduced_auc)

    best = defaultdict(list)
    for j, col in enumerate(LABEL_COLS + ['ALL']):
        xprint('#' * 100)
        method = 'MAX' if do_max else 'MEAN'
        xprint('RESULTS SUMMARY: %d - %d:%s %s %d' % (len(clf_auc), j, col, method, n_runs))
        if col == 'ALL':
            clf_order = sorted(clf_auc, key=lambda k: -clf_auc[k][1].mean())
        else:
            clf_order = sorted(clf_auc, key=lambda k: -clf_auc[k][1][j])
        clf0 = clf_order[0]
        if col == 'ALL':
            best[clf0].append((col, clf_auc[clf0][1].mean()))
        else:
            best[clf0].append((col, clf_auc[clf0][1][j]))
        # q, p = [clf_auc[clf][1] for clf in clf_order[:2]]
        # d = q - p
        # assert d.any() > 1e-4, (q, p, clf_order[:2])
        for i, clf in enumerate(clf_order[:2]):
            n_runs, auc = clf_auc[clf]
            xprint('auc=%.4f %3d: %s %s' % (auc.mean(), i, auc, clf))

    return best


def process_summary(path):
    print('path=%s' % path)
    completed_tests = load_json(path)
    xprint('run_summary_path=%s' % path)
    best = {}
    try:
        best = display_results(completed_tests, False)
        # display_results(completed_tests, True)
    except Exception as e:

        print('Bad summary: %s' % e)
    print('&' * 100)
    return best


xprint_init('consider_results', False)

path = 'logs/instance5/spacy_lstm19.9000.run_summary.json'
process_summary(path)

for summary_id in (3, 4, 5):
    print('summary_id=%d' % summary_id)
    summmary_dir = 'logs/instance%d' % summary_id
    summary_paths = glob(os.path.join(summmary_dir, '*.run_summary.json'))
    all_best = defaultdict(list)
    for path in summary_paths:
        best = process_summary(path)
        for k, v in best.items():
            all_best[k].extend(v)

    print('!' * 100)
    for i, k in enumerate(sorted(all_best)):
        print('%3d %10s: %s' % (i, k, all_best[k]))

    columns = sorted({c for w in all_best.values() for c, v in w})
    col_best = {}
    for col in columns:
        best = [(v, k) for k, w in all_best.items() for c, v in w if c == col]
        best.sort(key=lambda x: -x[0])
        col_best[col] = best

    print('#' * 100)
    for col in columns:
        print(col)
        for i, (v, k) in enumerate(col_best[col]):
            print('%3d %.4f: %s' % (i, v, k))
