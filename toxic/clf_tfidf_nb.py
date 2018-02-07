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
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from utils import COMMENT


# ## Building the model
#
# We'll start by creating a *bag of words* representation, as a *term document matrix*. We'll use
# ngrams, as suggested in the NBSVM paper.
import re
import string
from spacy_glue import SpacyCache


# do_spacy = False


# def set_spacy_tokenization(on):
#     global do_spacy
#     do_spacy = on


def get_tokenizer(do_spacy):
    if do_spacy:
        spacy_cache = SpacyCache()

        def tokenize(s):
            return spacy_cache.tokenize(s)[0]

    else:
        re_tok = re.compile('([%s“”¨«»®´·º½¾¿¡§£₤‘’])' % string.punctuation)

        def tokenize(s):
            """Surround punctuation with spaces then spilt on spaces"""
            return re_tok.sub(r' \1 ', s).split()

    return tokenize


if False:
    nlp = spacy.load('en')


    def tokenize1(s):
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


# Here's the basic naive bayes feature equation:
def pr(y_i, x, y):
    """
        y_i:  0 or 1
        y: the y vector
        returns: 1 x m vector  (x is an n x m vector)
                 y_pred[1:m]
                 y_pred[j] = sum p[:,j] / sum(p[:,:]) where p = x[y == y_i]

    """
    x_i = x[y == y_i]
    return (x_i.sum(0) + 1) / (x_i.sum() + 1)


# Fit a model for one dependent at a time:
def get_mdl(x, y):
    y = y.values
    r = np.log(pr(1, x, y) / pr(0, x, y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    print('** x y r x_nb:', x.shape, y.shape, r.shape, x_nb.shape)
    return m.fit(x_nb, y), r


class ClfTfidfNB:

    def __init__(self, label_cols, do_spacy):
        self.label_cols = label_cols
        print('TfidfVectorizer')
        t0 = time.clock()
        self.vec = TfidfVectorizer(ngram_range=(1, 2), tokenizer=get_tokenizer(do_spacy),
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1)
        print('TfidfVectorizer took %.1f seconds' % (time.clock() - t0))
        self.m = {}
        self.r = {}

    def fit(self, train):
        print('fit_transform')
        t0 = time.clock()
        x = self.vec.fit_transform(train[COMMENT])
        print('fit_transform took %.1f seconds' % (time.clock() - t0))
        for i, j in enumerate(self.label_cols):
            print('fit', j)
            self.m[j], self.r[j] = get_mdl(x, train[j])

    def predict(self, test):
        test_x = self.vec.transform(test[COMMENT])
        preds = np.zeros((len(test), len(self.label_cols)))
        for i, j in enumerate(self.label_cols):
            print('predict', j)
            m, r = self.m[j], self.r[j]
            preds[:, i] = m.predict_proba(test_x.multiply(r))[:, 1]
        return preds


