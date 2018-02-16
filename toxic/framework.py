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
from os.path import join
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
from utils import COMMENT, DATA_ROOT, dim, xprint


VERBOSE = False
GRAPHS = False
N_SAMPLES = 10000 # > 0 for testing
SEED = 234

SUBMISSION_DIR = 'submissions'
MODEL_DIR = 'models'
TOXIC_DATA_DIR = join(DATA_ROOT, 'toxic')
LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
seed_delta = 1


def seed_random(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def my_shuffle(indexes):
    global seed_delta

    old_state = random.getstate()
    random.seed(SEED + seed_delta)
    random.shuffle(indexes)
    random.setstate(old_state)
    seed_delta += 1


def load_data():
    train = pd.read_csv(join(TOXIC_DATA_DIR, 'train.csv'))
    test = pd.read_csv(join(TOXIC_DATA_DIR, 'test.csv'))
    subm = pd.read_csv(join(TOXIC_DATA_DIR, 'sample_submission.csv'))
    xprint('train,test,subm:', train.shape, test.shape, subm.shape)

    if N_SAMPLES > 0:
        train = train[:N_SAMPLES]
        test = test[:N_SAMPLES]

    seed_random()

    xprint('train=%d test=%d (%.1f%%)' % (len(train), len(test), 100.0 * len(test) / len(train)))

    # There are a few empty comments that we need to get rid of, otherwise sklearn will complain.
    train[COMMENT].fillna('_na_', inplace=True)
    test[COMMENT].fillna('_na_', inplace=True)

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


def split_data(df, frac):
    show_values('df', df)

    indexes = list(df.index)
    my_shuffle(indexes)
    n = int(len(df) * frac)
    train = df.loc[indexes[:n]]
    test = df.loc[indexes[n:]]

    show_values('train', train)
    show_values('test', test)
    xprint('split_data: %.2f of %d: train=%d test=%d' % (frac, len(df), len(train), len(test)))
    return train, test


def make_submission(get_clf, submission_name):
    submission_path = join(SUBMISSION_DIR, '%s.csv' % submission_name)
    assert not os.path.exists(submission_path), submission_path
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    train, test, subm = load_data()
    clf = get_clf()
    clf.fit(train)
    preds = clf.predict(test)

    # And finally, create the submission file.
    submid = pd.DataFrame({'id': subm['id']})
    submission = pd.concat([submid, pd.DataFrame(preds, columns=LABEL_COLS)], axis=1)
    submission.to_csv(submission_path, index=False)
    xprint('Saved in %s' % submission_path)


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



def _evaluate(get_clf, train, i, frac):
    xprint('_evaluate %3d %s' % (i, '-' * 66))
    train_part, test_part = split_data(train, frac)

    clf = get_clf()
    clf.fit(train_part)
    preds = clf.predict(test_part)

    auc = np.zeros(len(LABEL_COLS), dtype=np.float64)
    for j, col in enumerate(LABEL_COLS):
        y_true = test_part[col]
        y_pred = preds[:, j]
        auc[j] = roc_auc_score(y_true, y_pred)
    mean_auc = auc.mean()
    xprint('%5d: auc=%.3f %s' % (i, mean_auc, label_score(auc)))
    describe(preds)
    return auc


def auc_score(auc):
    mean_auc = auc.mean(axis=0)
    return mean_auc.mean(), mean_auc


def show_auc(auc):
    n = auc.shape[0]
    mean_auc = auc.mean(axis=0)

    xprint('-' * 110)
    for i in range(n):
        print('%5d: auc=%.3f %s' % (i, auc[i].mean(), label_score(auc[i])))
    xprint('%5s: auc=%.3f %s' % ('Mean', mean_auc.mean(), label_score(mean_auc)))
    xprint('-' * 110)
    xprint('auc=%.3f +- %.3f (%.0f%%) range=%.3f (%.0f%%)' % (
         mean_auc.mean(), mean_auc.std(),
         100.0 * mean_auc.std() / mean_auc.mean(),
         mean_auc.max() - mean_auc.min(),
         100.0 * (mean_auc.max() - mean_auc.min()) / mean_auc.mean()
    ))


def evaluate(get_clf, n=1, frac=0.7):
    train, _, _ = load_data()
    auc = np.zeros((n, len(LABEL_COLS)), dtype=np.float64)
    for i in range(n):
        auc[i, :] = _evaluate(get_clf, train, i, frac)
        show_auc(auc[:i + 1, :])
    xprint('program=%s' % sys.argv[0])
    return auc


if __name__ == '__main__':
    train, test, subm = load_data()
    # ## Looking at the data
    #
    # The training data contains a row per comment, with an id, the text of the comment, and 6 different
    # labels that we'll try to predict.
    if VERBOSE:
        print(train.head())

        # Here's a couple of examples of comments, one toxic, and one with no labels.
        print(train['comment_text'][0])
        print(train['comment_text'][2])

        # The length of the comments varies a lot.
        lens = train.comment_text.str.len()
        print(lens.mean(), lens.std(), lens.max())

    if GRAPHS:
        lens.hist()
        plt.show()

    # We'll create a list of all the labels to predict, and we'll also create a 'none' label so we can
    # see how many comments have no labels. We can then summarize the dataset.
    train['none'] = 1 - train[LABEL_COLS].max(axis=1)
    if VERBOSE:
        print('-' * 80)
        print('train')
        print(train.describe())
