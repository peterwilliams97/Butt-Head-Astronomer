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
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
from utils import COMMENT, DATA_ROOT, dim, xprint, load_json, save_json


VERBOSE = False
GRAPHS = False
N_SAMPLES = 10000  # > 0 for testing
SEED = 1111

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


def show_auc(auc):
    n = auc.shape[0]
    mean_auc = auc.mean(axis=0)
    auc_mean = auc.mean(axis=1)

    xprint('-' * 110, 'n=%d' % n)
    for i in range(n):
        print('%5d: auc=%.3f %s' % (i, auc[i].mean(), label_score(auc[i])))
    xprint('%5s: auc=%.3f %s' % ('Mean', mean_auc.mean(), label_score(mean_auc)))
    xprint('-' * 110)
    xprint('auc=%.3f +- %.3f (%.0f%%) range=%.3f (%.0f%%)' % (
         auc_mean.mean(), auc_mean.std(),
         100.0 * auc_mean.std() / auc_mean.mean(),
         auc_mean.max() - auc_mean.min(),
         100.0 * (auc_mean.max() - auc_mean.min()) / auc_mean.mean()
    ))


def cmt(row, m=120):
    text = row['comment_text'].strip()
    return '%3d %s' % (len(text), text[:m].replace('\n', ' '))


def show_best_worst(test, pred, n=20, m=100):
    pred_df = pd.DataFrame(pred, index=test.index, columns=['%s_pred' % col for col in LABEL_COLS])
    test_pred = pd.concat([test, pred_df], axis=1)
    print('test=%s pred=%s test_pred=%s' % (dim(test), dim(pred), dim(test_pred)))
    assert len(test) == len(pred) == len(test_pred)

    xprint('$' * 80)
    xprint('Best and worst predictions')
    for j, col in enumerate(LABEL_COLS):
        print('%2d: %-10s: %s' % (j, col, '`' * 80))
        col_pred = '%s_pred' % col
        test_pred.sort_values(col_pred, ascending=True, inplace=True)
        is_0 = test_pred[test_pred[col] == 0]
        is_1 = test_pred[test_pred[col] == 1]
        print('col=%s is_0=%d is_1=%d=%.1f%% ' % (col, len(is_0), len(is_1),
            100.0 * len(is_1) / len(test)))
        assert len(is_0) + len(is_1) == len(test_pred)
        # Best is_0, is_1
        xprint('Is 0. Predicted ~0. Good.', col)
        for i in range(min(len(is_0), n)):
            xprint('%4d: %.3f %s' % (i, is_0.iloc[i][col_pred], cmt(is_0.iloc[i])))
        xprint('Is 1. Predicted ~1. Good.', col)
        for i in range(min(len(is_1), n)):
            k = len(is_1) - 1 - i
            xprint('%4d: %.3f %s' % (i, is_1.iloc[k][col_pred], cmt(is_1.iloc[k])))
        # Worst is_0, is_1
        xprint('Is 1. Predicted ~0. BAD.', col)
        for i in range(min(len(is_1), n)):
            xprint('%4d: %.3f %s' % (i, is_1.iloc[i][col_pred], cmt(is_1.iloc[i])))
        xprint('Is 0. Predicted ~1. BAD.', col)
        for i in range(min(len(is_0), n)):
            k = len(is_0) - 1 - i
            xprint('%4d: %.3f %s' % (i, is_0.iloc[k][col_pred], cmt(is_0.iloc[k])))


class Evaluator:

    def __init__(self, n=1, frac=0.8):
        self.n = n
        self.frac = frac
        self.train, _, _ = load_data()
        self.shuffled_indexes = make_shuffled_indexes(self.train, self.n)
        assert len(self.shuffled_indexes) == n, (len(self.shuffled_indexes))

    def evaluate(self, get_clf):
        auc = np.zeros((self.n, len(LABEL_COLS)), dtype=np.float64)
        for i in range(self.n):
            ok, auc[i, :] = self._evaluate(get_clf, i)
            if not ok:
                return ok, auc
            show_auc(auc[:i + 1, :])
        xprint('program=%s train=%s' % (sys.argv[0], dim(self.train)))
        return True, auc

    def _evaluate(self, get_clf, i, do_clips=False):
        xprint('_evaluate %3d of %d  %s' % (i, self.n, '-' * 66))
        assert 0 <= i < len(self.shuffled_indexes), (i, self.n, len(self.shuffled_indexes))
        train_part, test_part = split_data(self.train, self.shuffled_indexes[i], self.frac)

        CLIPS = [0.0, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9]
        auc = np.zeros(len(LABEL_COLS), dtype=np.float64)

        try:
            clf = get_clf()
            clf.fit(train_part)
            pred = clf.predict(test_part)
            print('!!! pred=%s' % dim(pred))
        except Exception as e:
            xprint('!!! _evaluate, exception=%s' % e)
            raise
            return False, auc

        if do_clips:
            for k, delta in enumerate(CLIPS):
                auc = np.zeros(len(LABEL_COLS), dtype=np.float64)
                for j, col in enumerate(LABEL_COLS):
                    y_true = test_part[col]
                    y_pred = np.clip(pred[:, j], 0.0, 1.0 - delta)
                    auc[j] = roc_auc_score(y_true, y_pred)
                mean_auc = auc.mean()
                xprint('%5d: %d: delta=%6g auc=%.5f %s' % (i, k, delta, mean_auc, label_score(auc)))

        auc = np.zeros(len(LABEL_COLS), dtype=np.float64)
        for j, col in enumerate(LABEL_COLS):
            y_true = test_part[col]
            y_pred = pred[:, j]
            auc[j] = roc_auc_score(y_true, y_pred)
        mean_auc = auc.mean()
        xprint('%5d: auc=%.3f %s' % (i, mean_auc, label_score(auc)))
        describe(pred)
        show_best_worst(test_part, pred)
        return True, auc


def make_submission(get_clf, submission_name):
    submission_path = join(SUBMISSION_DIR, '%s.csv' % submission_name)
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


CHAR_COUNT = 'char_count.json'


def find_all_chars(verbose=False):

    char_count = load_json(CHAR_COUNT, {})
    if not char_count:
        train, test, _ = load_data()
        char_count = defaultdict(int)
        S = 0
        for df in (test, train):
            for sentence in df_to_sentences(df):
                S += 1
                for c in sentence:
                    assert len(c) == 1, (c, sentence)
                    char_count[c] += 1
        char_count = {c: n for c, n in char_count.items()}
        save_json(CHAR_COUNT, char_count)
        print('S=%d' % S)

    chars = sorted(char_count, key=lambda c: (-char_count[c], c))
    N = sum(char_count.values())
    print('find_all_chars: %d %r' % (len(chars), ''.join(chars[:100])))
    print('N=%d=%.3fM' % (N, N * 1e-6))

    if verbose:
        tot = 0.0
        for i, c in enumerate(chars[:200]):
            n = char_count[c]
            r = n / N
            tot += r
            print('%4d: %8d %.4f %.3f %4d=%2r' % (i, n, r, tot, ord(c), c))

    return char_count


if __name__ == '__main__':
    find_all_chars()
    assert False

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
