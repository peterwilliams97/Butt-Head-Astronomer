# coding: utf-8
"""
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
import time
from utils import COMMENT
from framework import LABEL_COLS
from spacy_glue import SpacyCache

if False:
    x = np.ones((3, 5))
    t = np.ones(5)
    x_nb = t * x
    x_nb = np.multiply
    assert False

SPACY_VECTOR_SIZE = 300  # To match SpaCY vectors
SPACY_VECTOR_TYPE = np.float32
SAVE_TIME = 60.0    # Save every 60 secs
SAVE_ITEMS = 2000   # Save if number of items in cache increases by this much


# Here's the basic naive bayes feature equation:
def pr(y_i, x, y):
    """
        y_i:  0 or 1
        y: the y vector
        returns: y_pred[1, m] 1 x m vector (x is an n x m vector)
                 y_pred[j] = sum p[:,j] / sum(p[:,:]) where p = x[y == y_i]

    """
    x_i = x[y == y_i]
    return (x_i.sum(0) + 1) / (x_i.sum() + 1)


def fit_model(x, y):
    """Fit a model for one dependent at a time
    """
    y = y.values
    t = np.zeros(y.shape[0], dtype=np.float64)
    y0 = y == 0
    y1 = y == 1
    t[y0] = -1.0
    t[y1] = 1.0
    # print('** y0, y1, t:', y0.shape, y1.shape, t.shape)
    # print('** x0, x1:', x0.shape, x1.shape)
    x_nb = (t * x.T).T
    # print('** x_nb:', x_nb.shape)
    # x_nb = x_nb.reshape(-1, 1)
    # print('** x_nb:', x_nb.shape)

    # print('** x, r, x_nb, y:', x.shape, r.shape, x_nb.shape, y.shape)
    m = LogisticRegression(C=4, dual=True)
    return m.fit(x_nb, y)


class ClfGloveNBSpacy:

    def __init__(self, n_gram):
        self.LABEL_COLS = LABEL_COLS
        self.n_gram = n_gram

        self.spacy_cache = SpacyCache()
        self.nlp = self.spacy_cache.nlp
        self.m = {}
        self.r = {}

    def tokenize_df(self, df):
        """Return a list of token lists
            Each token list corresponds to a row of df
        """
        print('tokenize_df: %d %d' % (len(df), self.spacy_cache.text_tokens_len))
        t0 = t1 = time.clock()
        tokens_list = []
        n_tokenized = 0
        for i, text in enumerate(df[COMMENT]):
            tokens, loaded = self.spacy_cache.tokenize(text)
            n_tokenized += loaded
            tokens_list.append(tokens)
            t = time.clock()
            if t > t1 + SAVE_TIME or i + 1 == len(df):
                print('tokenize_df: %d texts, %d new, %1.f sec (%.2f sec / token)' % (
                    i + 1, n_tokenized,
                    time.clock() - t0,
                    (time.clock() - t0) / max(n_tokenized, 1)))
                t1 = t
                self.spacy_cache._save(min_delta=SAVE_ITEMS)
        self.spacy_cache._save()
        return tokens_list

    def compute_ngram_vector(self, token_list, n):
        """Compute an embedding vector for all n-grams in token_list
        """
        token_vector = self.spacy_cache.token_vector
        vec = np.zeros((n, SPACY_VECTOR_SIZE), dtype=np.float64)
        n_toks = len(token_list) - n + 1
        if n_toks <= 0:
            for j in range(len(token_list)):
                vec[j, :] += token_vector[token_list[j]]
        else:
            for i in range(n_toks):
                for j in range(n):
                    vec[j, :] += token_vector[token_list[i + j]]
            vec /= n_toks
        return np.reshape(vec, n * SPACY_VECTOR_SIZE)

    def compute_ngram_matrix(self, token_matrix):
        mat = np.zeros((len(token_matrix), self.n_gram * SPACY_VECTOR_SIZE), dtype=np.float64)
        for i, token_list in enumerate(token_matrix):
            mat[i, :] = self.compute_ngram_vector(token_list, self.n_gram)
        return mat

    def fit(self, train):
        t0 = time.clock()
        train_tokens = self.tokenize_df(train)
        print('train_tokens: %1.f sec %.2f sec / token' % (time.clock() - t0, (time.clock() - t0) / len(train_tokens)))
        x = self.compute_ngram_matrix(train_tokens)
        for i, col in enumerate(self.LABEL_COLS):
            self.m[col] = fit_model(x, train[col])

    def predict(self, test):
        LABEL_COLS = self.LABEL_COLS
        preds = np.zeros((len(test), len(LABEL_COLS)))
        test_tokens = self.tokenize_df(test)
        test_x = self.compute_ngram_matrix(test_tokens)
        for i, col in enumerate(self.LABEL_COLS):
            # print('fit', i, col)
            m = self.m[col]
            preds[:, i] = m.predict_proba(test_x)[:, 1]
        return preds
