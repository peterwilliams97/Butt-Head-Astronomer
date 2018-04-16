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
from sklearn.model_selection import train_test_split, KFold
from keras import backend as K
from keras.layers import GRU, LSTM, CuDNNGRU, CuDNNLSTM
# import matplotlib.pyplot as plt
from utils import COMMENT, DATA_ROOT, dim, xprint


VERBOSE = False
GRAPHS = False
_N_SAMPLES = [-1]  # > 0 for testing
SEED = 234

SUBMISSION_DIR = 'submissions.cv'
MODEL_DIR = 'models.cv'
TOXIC_DATA_DIR = join(DATA_ROOT, 'toxic')
SUMMARY_DIR = 'run.summaries'
seed_delta = 1

os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)


gpu_list = K.tensorflow_backend._get_available_gpus()
xprint('gpu_list=%s' % gpu_list)
if gpu_list:
    K_GRU = CuDNNGRU
    K_LSTM = CuDNNLSTM
else:
    K_GRU = GRU
    K_LSTM = LSTM
xprint('K_GRU=%s' % K_GRU.__name__)
xprint('K_LSTM=%s' % K_LSTM.__name__)


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


def _df_split(df, frac):
    test_size = 1.0 - frac
    y = df[LABEL_COLS].values
    X = list(df.index)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y)
    train = df.loc[X_train]
    test = df.loc[y_train]
    return train, test


# def split_data(df, indexes, frac):
#     show_values('df', df)

#     n = int(len(df) * frac)
#     train = df.loc[indexes[:n]]
#     test = df.loc[indexes[n:]]

#     show_values('train', train)
#     show_values('test', test)
#     xprint('split_data: %.2f of %d: train=%d test=%d' % (frac, len(df), len(train), len(test)))
#     return train, test


def label_score(auc):
    return '(%s)' % ', '.join(['%s:%.3f' % (col, auc[j]) for j, col in enumerate(LABEL_COLS)])


def auc_score(auc):
    mean_auc = auc.mean(axis=0)
    return mean_auc.mean(), mean_auc


def auc_score_list(auc):
    mean_mean, mean_auc = auc_score(auc)
    return [mean_mean, mean_auc.tolist()]


def show_auc(auc):
    assert len(auc.shape) == 2, dim(auc)
    n = auc.shape[0]
    mean_auc = auc.mean(axis=0)
    auc_mean = auc.mean(axis=1)

    xprint('-' * 110, 'n=%d' % n)
    for i in range(n):
        xprint('%5d: auc=%.4f %s' % (i, auc[i, :].mean(), label_score(auc[i, :])))
    xprint('%5s: auc=%.4f %s' % ('Mean', mean_auc.mean(), label_score(mean_auc)))
    xprint('-' * 110)
    xprint('auc=%.4f +- %.4f (%.2f%%) range=%.3f (%.2f%%)' % (
         auc_mean.mean(), auc_mean.std(),
         100.0 * auc_mean.std() / auc_mean.mean(),
         auc_mean.max() - auc_mean.min(),
         100.0 * (auc_mean.max() - auc_mean.min()) / auc_mean.mean()
    ))


def show_results(auc_list):
    """auc_list: list of auc, clf, clf_str
    """
    results = [(i, auc, best_epoch, best_auc, dt_fit, dt_pred, clf, clf_str) for i, (auc, best_epoch, best_auc,
        dt_fit, dt_pred, clf, clf_str) in enumerate(auc_list)]
    results.sort(key=lambda x: (-x[1].mean(), x[2], x[3]))
    xprint('~' * 100)
    xprint('RESULTS SUMMARY: %d' % len(results))
    for i, auc, best_epoch, best_auc, dt_fit, dt_pred, clf, clf_str in results:
        xprint('auc=%.4f %3d: %s %s best_epoch=%d best_auc=%.4f dt_fit=%.1f sec dt_pred=%.1f sec' % (
            auc.mean(), i, clf, clf_str, best_epoch, best_auc, dt_fit, dt_pred))


def show_results_cv(auc_list):
    """auc_list: list of auc, clf, clf_str
    """
    results = [(i, auc, clf_str) for i, (auc, clf_str) in enumerate(auc_list)]
    results.sort(key=lambda x: -x[1].mean())
    xprint('~' * 100)
    xprint('RESULTS SUMMARY: %d' % len(results))
    for i, auc, clf_str in results:
        xprint('auc=%.4f %3d: %s' % (auc.mean(), i, clf_str))


LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def _split(df, validation_size):
    y = df[LABEL_COLS].values
    X = list(df.index)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_size, stratify=n)
    train = df.loc[X_train]
    test = df.loc[y_train]
    return train, test


def prepare_data(validation_size=0.0):
    train, test, subm = load_data()

    valid = None
    if validation_size:
        train, valid = train_test_split(train, test_size=validation_size)

    idx_train = train.index
    X_train = train["comment_text"].values
    y_train = train[LABEL_COLS].values
    if valid is not None:
        idx_test = valid.index
        X_test = valid["comment_text"].values
        y_test = valid[LABEL_COLS].values
    else:
        idx_test = test.index
        X_test = test["comment_text"].values
        y_test = None

    xprint('prepare_data: X_train=%s y_train=%s' % (dim(X_train), dim(y_train)))
    xprint('prepare_data: X_test=%s y_test=%s' % (dim(X_test), dim(y_test)))

    return (idx_train, X_train, y_train), (idx_test, X_test, y_test)


def calc_auc(y_true_all, y_pred_all):
    auc = np.zeros(len(LABEL_COLS), dtype=np.float64)
    for j, col in enumerate(LABEL_COLS):
        y_true = y_true_all[:, j]
        y_pred = y_pred_all[:, j]
        auc[j] = roc_auc_score(y_true, y_pred)
    mean_auc = auc.mean()
    xprint('auc=%.4f %s' % (mean_auc, label_score(auc)))
    return auc


def evaluate(get_clf, X_valid, y_valid):
    clf = get_clf()
    xprint('^' * 100)
    xprint('evaluate: clf=%s' % str(clf))
    clf.show_model()
    xprint('#' * 100)

    y_pred = clf.predict(X_valid)
    auc = calc_auc(y_valid, y_pred)
    score = auc.mean()

    xprint('score=%.4f' % score)
    return auc


class CV_predictor():
    """
        class to extract predictions on train and test set from tuned pipeline
    """

    def __init__(self, get_clf, idx_train, x_train, y_train, idx_test, x_test, y_test, n_splits,
        batch_size, epochs):
        self.get_clf = get_clf
        self.cv = KFold(n_splits=n_splits, shuffle=True, random_state=1)
        self.idx_train = idx_train
        self.x_train = x_train
        self.y_train = y_train
        self.idx_test = idx_test
        self.x_test = x_test
        self.y_test = y_test

        self.epochs = epochs
        self.batch_size = batch_size
        self.col_names = LABEL_COLS

    def predict(self):
        self.train_predictions = []
        self.test_predictions = np.zeros((self.cv.get_n_splits(), self.x_test.shape[0], len(LABEL_COLS)))
        self.auc_train = np.zeros((self.cv.get_n_splits(), len(LABEL_COLS)))

        for cv_i, (train_i, valid_i) in enumerate(self.cv.split(self.x_train, self.y_train)):
            clf = self.get_clf()
            clf_str = str(clf)
            if cv_i == 0:
                xprint('^' * 100)
                xprint('evaluate: clf=%s' % str(clf))
                clf.show_model()
            xprint('#' * 100)
            xprint('evaluate: CV round %d of %d' % (cv_i + 1, self.cv.get_n_splits()))

            x_train = self.x_train[train_i]
            y_train = self.y_train[train_i]
            x_valid = self.x_train[valid_i]
            y_valid = self.y_train[valid_i]

            scores = np.zeros(self.epochs)
            auc_train = np.zeros((self.epochs, len(LABEL_COLS)))
            best_i = -1
            best_score = 0.0
            for i in range(self.epochs):
                xprint('epoch %d of %d %s' % (i + 1, self.epochs, '+' * 80))
                clf.fit(x_train, y_train)
                train_prediction = clf.predict(x_valid)

                auc = calc_auc(y_valid, train_prediction)
                auc_train[i, :] = auc
                scores[i] = score = auc.mean()
                best_i = np.argmax(scores)
                best_score = scores[best_i]

                xprint('CV=%d epoch=%d (%d) score=%.4f (%.4f)' % (cv_i + 1, i + 1, best_i + 1, score,
                    best_score))
                xprint('scores=%s' % scores[:i + 1].T)
                xprint(auc_train[:i + 1, :])

            xprint('\/' * 40)
            xprint('CV round %d predict' % (cv_i + 1))
            y_pred = clf.predict(self.x_test)
            auc = calc_auc(y_valid, train_prediction)

            self.train_predictions.append([train_prediction, valid_i])
            self.test_predictions[cv_i, :, :] = y_pred
            self.auc_train[cv_i, :] = auc
            show_auc(self.auc_train[:cv_i + 1, :])
            del clf

        xprint('=' * 100)
        xprint('Done evaluate: clf=%s' % clf_str)
        xprint('All CVs: score=%s' % self.auc_train.mean(axis=1))
        xprint('All CVs: avg score=%.4f' % self.auc_train.mean())
        show_auc(self.auc_train)

        self.train_predictions = (
            pd.concat([pd.DataFrame(data=data, index=idx, columns=LABEL_COLS)
                       for data, idx in self.train_predictions]).sort_index())
        self.test_predictions = pd.DataFrame(data=np.mean(self.test_predictions, axis=0),
                                             index=self.idx_test, columns=LABEL_COLS)
        xprint('~' * 100)

        # print('test_predictions=%s x_test=%s' % (dim(self.test_predictions), dim(self.x_test)))
        assert len(self.test_predictions.shape) == 2, (dim(self.test_predictions), dim(self.x_test))
        assert self.test_predictions.shape[0] == self.x_test.shape[0], (dim(self.test_predictions), dim(self.x_test))

        return self.auc_train

    def eval_predictions(self):
        if self.y_test is None:
            xprint('eval_predictions: Needs y_test')
            return None
        test = pd.DataFrame(data=self.y_test, index=self.idx_test, columns=LABEL_COLS).sort_index()
        y_test = test[LABEL_COLS].values
        y_pred = self.test_predictions[LABEL_COLS].values
        auc = calc_auc(y_test, y_pred)
        xprint('eval_predictions: score=%.4f' % auc.mean())
        return score


def make_submission(get_clf, submission_name):
    seed_random()

    train, test, subm = load_data()
    clf = get_clf()
    clf.fit(train, test_size=0.0)
    pred_data = clf.predict(test)
    pred = pd.DataFrame(pred_data, columns=LABEL_COLS)
    save_submission(pred, submission_name)


def save_submission(pred, submission_name):

    _, _, subm = load_data()
    # Create the submission file.
    submid = pd.DataFrame({'id': subm['id']})
    print('submid=%s pred=%s' % (submid, pred))
    submission = pd.concat([submid, pred], axis=1)

    submission_path = join(SUBMISSION_DIR, submission_name)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_path = join(SUBMISSION_DIR, submission_name)
    if os.path.exists(submission_path):
        submission_dir_old = '%s.old' % SUBMISSION_DIR
        submission_path_old = join(submission_dir_old, submission_name)
        xprint('*** Renaming old submission %s->%s' % (submission_path, submission_path_old))
        os.makedirs(submission_dir_old, exist_ok=True)
        os.rename(submission_path, submission_path_old)

    submission.to_csv(submission_path, index=False)
    xprint('Saved in %s' % submission_path)
    xprint('program=%s submission=%s' % (sys.argv[0], dim(submission)))
