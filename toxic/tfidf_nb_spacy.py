# coding: utf-8
"""
    This kernel shows how to use NBSVM (Naive Bayes - Support Vector Machine) to create a strong
    baseline for the Toxic Comment Classification Challenge
    https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) competition.

    NBSVM was introduced by Sida Wang and Chris Manning in the paper Baselines and Bigrams: Simple,
    Good Sentiment and Topic Classiﬁcation (https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf).
    In this kernel, we use sklearn's logistic regression, rather than SVM, although in practice the
    two are nearly identical (sklearn uses the liblinear library behind the scenes).

    If you're not familiar with naive bayes and bag of words matrices, I've made a preview available
    of one of fast.ai's upcoming *Practical Machine Learning* course videos, which introduces this
    topic. Here is a link to the section of the video which discusses this:
    [Naive Bayes video](https://youtu.be/37sFIak42Sc?t=3745).
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from os.path import expanduser, join
import matplotlib.pyplot as plt
import spacy


data_dir = expanduser('/Users/pcadmin/data/toxic/')
train = pd.read_csv(join(data_dir, 'train.csv'))
test = pd.read_csv(join(data_dir, 'test.csv'))
subm = pd.read_csv(join(data_dir, 'sample_submission.csv'))


# ## Looking at the data
#
# The training data contains a row per comment, with an id, the text of the comment, and 6 different
# labels that we'll try to predict.
print(train.head())

# Here's a couple of examples of comments, one toxic, and one with no labels.
train['comment_text'][0]
train['comment_text'][2]


# The length of the comments varies a lot.
lens = train.comment_text.str.len()
print(lens.mean(), lens.std(), lens.max())

if False:
    lens.hist()
    plt.show()


# We'll create a list of all the labels to predict, and we'll also create a 'none' label so we can
# see how many comments have no labels. We can then summarize the dataset.
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1 - train[label_cols].max(axis=1)
train.describe()

print('train=%d test=%d (%.1f%%)' % (len(train), len(test), 100.0 * len(test) / len(train)))

# There are a few empty comments that we need to get rid of, otherwise sklearn will complain.
COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)

# ## Building the model
#
# We'll start by creating a *bag of words* representation, as a *term document matrix*. We'll use
# ngrams, as suggested in the NBSVM paper.
import re
import string


re_tok = re.compile('([%s“”¨«»®´·º½¾¿¡§£₤‘’])' % string.punctuation)


def tokenize0(s):
    """Surround punctuation with spaces then spilt on spaces"""
    return re_tok.sub(r' \1 ', s).split()


nlp = spacy.load('en')


def tokenize(s):
    """Surround punctuation with spaces then spilt on spaces"""
    doc = nlp(s)
    return [t.text for t in doc]


if False:
    s = '''It turns out "why is" that's using. Doesn't it? Can't i'''
    t0 = tokenize0(s)
    print(t0)
    t = tokenize(s)
    print(t)
    print(re_tok.pattern)
    assert False


# It turns out that using TF-IDF gives even better priors than the binarized features used in the
# paper. I don't think this has been mentioned in any paper before, but it improves leaderboard
# score from 0.59 to 0.55.
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])


# This creates a *sparse matrix* with only a small number of non-zero elements (*stored elements*
# in the representation  below).
print(trn_term_doc, test_term_doc)


# Here's the basic naive bayes feature equation:
def pr(y_i, y):
    """
        y_i:  0 or 1
        y: the y vector
        returns: 1 x m vector  (x is an n x m vector)
                 y_pred[1:m]
                 y_pred[j] = sum p[:,j] / sum(p[:,:]) where p = x[y == y_i]

    """
    x_i = x[y == y_i]
    return (x_i.sum(0) + 1) / (x_i.sum() + 1)


x = trn_term_doc
test_x = test_term_doc
type(test_x)


# Fit a model for one dependent at a time:
def get_mdl(y):
    y = y.values
    r = np.log(pr(1, y) / pr(0, y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r


preds = np.zeros((len(test), len(label_cols)))

for i, j in enumerate(label_cols):
    print('fit', j)
    m, r = get_mdl(train[j])
    preds[:, i] = m.predict_proba(test_x.multiply(r))[:, 1]

print()

# And finally, create the submission file.
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns=label_cols)], axis=1)
submission.to_csv('submission001.csv', index=False)
