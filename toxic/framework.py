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
from clf_glove_nb_spacy import ClfGloveNBSpace


VERBOSE = False
GRAPHS = False
N_GRAM = 1
N_SAMPLES = 50000  # > 0 for testing
MODEL = ClfGloveNBSpace
MODEL_DIR = 'model.%s' % MODEL.__name__
SUBMISSION_NAME = 'submissionxxx.csv'
SEED = 234

print('MODEL_DIR=%s' % MODEL_DIR)

data_dir = expanduser('/Users/pcadmin/data/toxic/')
train = pd.read_csv(join(data_dir, 'train.csv'))
test = pd.read_csv(join(data_dir, 'test.csv'))
subm = pd.read_csv(join(data_dir, 'sample_submission.csv'))
print('train,test,subm:', train.shape, test.shape, subm.shape)
os.makedirs(MODEL_DIR, exist_ok=True)

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


def split_data(df, frac):
    # indexes = list(range(len(df)))
    indexes = list(df.index)
    random.shuffle(indexes)
    n = int(len(df) * frac)
    train = df.loc[indexes[:n]]
    test = df.loc[indexes[n:]]
    return train, test


def make_submission(train, test):
    clf = MODEL(label_cols, MODEL_DIR, N_GRAM)
    clf.fit(train)
    preds = clf.predict(test)

    # And finally, create the submission file.
    submid = pd.DataFrame({'id': subm['id']})
    submission = pd.concat([submid, pd.DataFrame(preds, columns=label_cols)], axis=1)
    submission.to_csv(SUBMISSION_NAME, index=False)


def evaluate(train0):
    train, test = split_data(train0, 0.7)
    clf = MODEL(label_cols, MODEL_DIR, N_GRAM)
    clf.fit(train)
    preds = clf.predict(test)
    auc = np.zeros(len(label_cols), dtype=np.float64)
    for i, col in enumerate(label_cols):
        y_true = test[col]
        y_pred = preds[:, i]
        auc[i] = roc_auc_score(y_true, y_pred)
    mean_auc = auc.mean()
    print('auc=%.3f' % mean_auc)


# make_submission(train, test)
evaluate(train)
