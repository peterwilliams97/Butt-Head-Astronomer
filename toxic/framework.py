# coding: utf-8
"""
    Framework for evaluating and submitting models to Kaggle Toxic Comment challenge
"""
import numpy as np
import pandas as pd
import os
import random
from os.path import expanduser, join
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from utils import COMMENT


VERBOSE = False
GRAPHS = False
N_SAMPLES = -1  # > 0 for testing
SEED = 234
SUBMISSION_DIR = 'submissions'

data_dir = expanduser('~/data/toxic/')

if __name__ == '__main__':
    train = pd.read_csv(join(data_dir, 'train.csv'))
    test = pd.read_csv(join(data_dir, 'test.csv'))
    subm = pd.read_csv(join(data_dir, 'sample_submission.csv'))
    print('train,test,subm:', train.shape, test.shape, subm.shape)

    if N_SAMPLES > 0:
        train = train[:N_SAMPLES]
        test = test[:N_SAMPLES]

    random.seed(SEED)
    np.random.seed(SEED)

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
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train['none'] = 1 - train[label_cols].max(axis=1)
    if VERBOSE:
        print('-' * 80)
        print('train')
        print(train.describe())

    print('train=%d test=%d (%.1f%%)' % (len(train), len(test), 100.0 * len(test) / len(train)))

    # There are a few empty comments that we need to get rid of, otherwise sklearn will complain.
    train[COMMENT].fillna("unknown", inplace=True)
    test[COMMENT].fillna("unknown", inplace=True)

seed_delta = 1


def my_shuffle(indexes):
    global seed_delta

    old_state = random.getstate()
    random.seed(SEED + seed_delta)
    random.shuffle(indexes)
    random.setstate(old_state)
    seed_delta += 1


def split_data(df, frac):
    indexes = list(df.index)
    my_shuffle(indexes)
    n = int(len(df) * frac)
    train = df.loc[indexes[:n]]
    test = df.loc[indexes[n:]]
    print('split_data: %.2f of %d: train=%d test=%d' % (frac, len(df), len(train), len(test)))
    return train, test


def make_submission(get_clf, submission_name):
    clf = get_clf()
    clf.fit(train)
    preds = clf.predict(test)

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_path = join(SUBMISSION_DIR, submission_name)

    # And finally, create the submission file.
    submid = pd.DataFrame({'id': subm['id']})
    submission = pd.concat([submid, pd.DataFrame(preds, columns=label_cols)], axis=1)
    submission.to_csv(submission_path, index=False)
    print('Saved in %s' % submission_path)


def label_score(auc):
    return '(%s)' % ', '.join(['%s:%.3f' % (col, auc[j])
                              for j, col in enumerate(label_cols)])


def _evaluate(get_clf, i):
    train_part, test_part = split_data(train, 0.7)

    clf = get_clf()
    clf.fit(train_part)
    preds = clf.predict(test_part)

    auc = np.zeros(len(label_cols), dtype=np.float64)
    for j, col in enumerate(label_cols):
        y_true = test_part[col]
        y_pred = preds[:, j]
        auc[j] = roc_auc_score(y_true, y_pred)
    mean_auc = auc.mean()
    print('%5d: auc=%.3f %s' % (i, mean_auc, label_score(auc)))
    return auc


def evaluate(get_clf, n=5):
    auc = np.zeros((n, len(label_cols)), dtype=np.float64)
    for i in range(n):
        auc[i, :] = _evaluate(get_clf, i)
    mean_auc = auc.mean(axis=0)

    print('-' * 110)
    for i in range(n):
        print('%5d: auc=%.3f %s' % (i, auc[i].mean(), label_score(auc[i])))
    print('%5s: auc=%.3f %s' % ('Mean', mean_auc.mean(), label_score(mean_auc)))
    print('-' * 110)
    print('auc=%.3f +- %.3f (%.0f%%) range=%.3f (%.0f%%)' % (
         mean_auc.mean(), mean_auc.std(),
         100.0 * mean_auc.std() / mean_auc.mean(),
         mean_auc.max() - mean_auc.min(),
         100.0 * (mean_auc.max() - mean_auc.min()) / mean_auc.mean()
    ))
    return auc

