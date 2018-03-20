# coding: utf-8
"""
    Framework for evaluating and submitting models to Kaggle Toxic Comment challenge
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import random
import sys
import time
from os.path import join
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
from utils import COMMENT, DATA_ROOT, dim, xprint


VERBOSE = False
GRAPHS = False
_N_SAMPLES = [-1]  # > 0 for testing
SEED = 234

SUBMISSION_DIR = 'submissions.fasttext'
MODEL_DIR = 'models.fasttext'
TOXIC_DATA_DIR = join(DATA_ROOT, 'toxic')
SUMMARY_DIR = 'run.summaries'
seed_delta = 1

os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)


def set_n_samples(n):
    _N_SAMPLES[0] = n


def get_n_samples():
    return _N_SAMPLES[0]


def get_n_samples_str():
    n_samples = get_n_samples()
    return 'ALL' if n_samples < 0 else str(n_samples)


_random_seed = [SEED]


def seed_random(seed=None):
    if seed is None:
        seed = _random_seed[0]
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def set_random_seed(seed):
    xprint('set_random_seed: seed=%d' % seed)
    _random_seed[0] = seed
    seed_random()


def my_shuffle(indexes):
    global seed_delta

    old_state = random.getstate()
    random.seed(SEED + seed_delta)
    random.shuffle(indexes)
    random.setstate(old_state)
    seed_delta += 1


def load_base_data():
    print('load_base_data')
    train = pd.read_csv(join(TOXIC_DATA_DIR, 'train.csv'))
    test = pd.read_csv(join(TOXIC_DATA_DIR, 'test.csv'))
    subm = pd.read_csv(join(TOXIC_DATA_DIR, 'sample_submission.csv'))

    # There are a few empty comments that we need to get rid of, otherwise sklearn will complain.
    train[COMMENT].fillna('_na_', inplace=True)
    test[COMMENT].fillna('_na_', inplace=True)

    xprint('train,test,subm:', train.shape, test.shape, subm.shape)
    X_train = train[COMMENT].values
    X_test = test[COMMENT].values
    print('load_base_data: X_train=%s X_test=%s' % (dim(X_train), dim(X_test)))
    return train, test, subm, X_train, X_test


train0, test0, subm0, X_train0, X_test0 = load_base_data()


def load_data():
    train = train0
    test = test0
    subm = subm0
    xprint('train,test,subm:', train.shape, test.shape, subm.shape)

    n_samples = get_n_samples()
    if n_samples > 0:
        train = train[:n_samples]
        test = test[:n_samples]

    seed_random()

    xprint('train=%d test=%d (%.1f%%)' % (len(train), len(test), 100.0 * len(test) / len(train)))

    return train, test, subm


def df_to_sentences(df):
    assert not any(df['comment_text'].isnull())
    return df['comment_text'].fillna('_na_').values


def show_values(name, df):
    return False
    print('~' * 80)
    print('%10s: %6d rows' % (name, len(df)))
    for i, col in enumerate(LABEL_COLS):
        print('%3d: %s' % (i, col))
        print(df[col].value_counts())


def df_split(df, frac):
    test_size = 1.0 - frac
    y = df[LABEL_COLS].values
    X = list(df.index)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y)
    train = df.loc[X_train]
    test = df.loc[y_train]
    return train, test


def make_shuffled_indexes(df, n):

    indexes = list(df.index)
    shuffled_indexes = []
    for i in range(n):
        my_shuffle(indexes)
        shuffled_indexes.append(indexes[:])
    assert len(shuffled_indexes) == n
    return shuffled_indexes


def split_data(df, indexes, frac):
    show_values('df', df)

    n = int(len(df) * frac)
    train = df.loc[indexes[:n]]
    test = df.loc[indexes[n:]]

    show_values('train', train)
    show_values('test', test)
    xprint('split_data: %.2f of %d: train=%d test=%d' % (frac, len(df), len(train), len(test)))
    return train, test


def label_score(auc):
    return '(%s)' % ', '.join(['%s:%.3f' % (col, auc[j]) for j, col in enumerate(LABEL_COLS)])


def describe(y):
    """Return table of values
        min, mean, max
    """
    MEASURES = ['min', 'mean', 'max']
    stats = np.zeros((3, len(LABEL_COLS)), dtype=np.float64)
    xprint('stats=%s' % dim(stats))
    for j, col in enumerate(LABEL_COLS):
        stats[0, j] = y[:, j].min()
        stats[1, j] = y[:, j].mean()
        stats[2, j] = y[:, j].max()

    def draw(name, vals, sep='|'):
        vals = ['%12s' % v for v in ([name] + vals)]
        xprint((' %s ' % sep).join(vals))

    def draw_bar():
        bar = '-' * 12
        draw(bar, [bar] * len(LABEL_COLS), sep='+')

    draw_bar()
    draw('', LABEL_COLS)
    draw_bar()
    for i, measure in enumerate(MEASURES):
        draw(measure, ['%10.4f' % z for z in stats[i, :]])
    draw_bar()


if False:
    y = np.random.uniform(low=-1.0, high=1.0, size=(1000, len(LABEL_COLS)))
    print('y=%s' % dim(y))
    describe(y)
    assert False


def auc_score(auc):
    mean_auc = auc.mean(axis=0)
    return mean_auc.mean(), mean_auc


def auc_score_list(auc):
    mean_mean, mean_auc = auc_score(auc)
    return [mean_mean, mean_auc.tolist()]


def show_auc(auc):
    n = auc.shape[0]
    mean_auc = auc.mean(axis=0)
    auc_mean = auc.mean(axis=1)

    xprint('-' * 110, 'n=%d' % n)
    for i in range(n):
        xprint('%5d: auc=%.3f %s' % (i, auc[i, :].mean(), label_score(auc[i, :])))
    xprint('%5s: auc=%.3f %s' % ('Mean', mean_auc.mean(), label_score(mean_auc)))
    xprint('-' * 110)
    xprint('auc=%.3f +- %.3f (%.0f%%) range=%.3f (%.0f%%)' % (
         auc_mean.mean(), auc_mean.std(),
         100.0 * auc_mean.std() / auc_mean.mean(),
         auc_mean.max() - auc_mean.min(),
         100.0 * (auc_mean.max() - auc_mean.min()) / auc_mean.mean()
    ))


def show_results(auc_list):
    """auc_list: list of auc, clf, clf_str
    """
    results = [(i, auc, clf, clf_str) for i, (auc, clf, clf_str) in enumerate(auc_list)]
    results.sort(key=lambda x: (-x[1].mean(), x[2], x[3]))
    xprint('~' * 100)
    # xprint('RESULTS SO FAR: %d' % len(results))
    # for i, auc, clf, clf_str in results:
    #     xprint('$' * 100)
    #     xprint('auc=%.4f %3d: %s %s' % (auc.mean(), i, clf, clf_str))
    #     show_auc(auc)
    xprint('^' * 100)
    xprint('RESULTS SUMMARY: %d' % len(results))
    for i, auc, clf, clf_str in results:
        xprint('auc=%.4f %3d: %s %s' % (auc.mean(), i, clf, clf_str))


LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


class Evaluator:

    def __init__(self, frac=0.8):
        self.frac = frac
        self.train, _, _ = load_data()
        self.shuffled_indexes = make_shuffled_indexes(self.train, 1)
        seed_random()

    def evaluate(self, clf_str, get_clf, *args, **keywords):

        clf = get_clf(*args, **keywords)
        self.clf_ = clf
        xprint('evaluate: clf    =%s' % str(clf))
        xprint('evaluate: clf_str=%s' % clf_str)
        assert str(clf) == clf_str

        train_part, test_part = split_data(self.train, self.shuffled_indexes[0], self.frac)
        X_train = train_part["comment_text"].values
        y_train = train_part[LABEL_COLS].values
        X_test = test_part["comment_text"].values
        y_test = test_part[LABEL_COLS].values
        xprint('evaluate: X_train=%s y_train=%s' % (dim(X_train), dim(y_train)))
        xprint('evaluate: X_test=%s y_test=%s' % (dim(X_test), dim(y_test)))
        # assert len(X_train) >= 20000
        # assert len(X_test) >= 20000


        auc = np.zeros(len(LABEL_COLS), dtype=np.float64)
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        xprint('evaluate fit duration=%.1f sec' % (time.perf_counter() - t0))
        t0 = time.perf_counter()
        pred = clf.predict(X_test)
        xprint('evaluate predict duration=%.1f sec' % (time.perf_counter() - t0))
        xprint('y_test=%s pred=%s' % (dim(y_test), dim(pred)))

        auc = np.zeros(len(LABEL_COLS), dtype=np.float64)
        for j, col in enumerate(LABEL_COLS):
            xprint('^^ %d: %s ' % (j, col), end=' ')
            y_true = y_test[:, j]
            y_pred = pred[:, j]
            xprint('y_true=%s y_pred=%s' % (dim(y_true), dim(y_pred)))
            auc[j] = roc_auc_score(y_true, y_pred)
        mean_auc = auc.mean()
        xprint('auc=%.3f %s' % (mean_auc, label_score(auc)))
        describe(pred)

        if clf is not None:
            del clf
        return auc


def make_submission(get_clf, submission_name):
    seed_random()
    submission_path = join(SUBMISSION_DIR, '%s.%s.csv' % (submission_name, get_n_samples_str()))
    assert not os.path.exists(submission_path), submission_path
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    train, test, subm = load_data()
    clf = get_clf()
    clf.fit(train, test_size=0.0)
    pred = clf.predict(test)

    describe(pred)

    # Csreate the submission file.
    submid = pd.DataFrame({'id': subm['id']})
    submission = pd.concat([submid, pd.DataFrame(pred, columns=LABEL_COLS)], axis=1)
    submission.to_csv(submission_path, index=False)
    xprint('Saved in %s' % submission_path)
    xprint('program=%s train=%s test=%s submission=%s' % (sys.argv[0], dim(train), dim(test),
        dim(submission)))
