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
import os
import spacy
import time
from utils import COMMENT, save_json, load_json, save_pickle, load_pickle


N_GRAM = 3
SPACY_VECTOR_SIZE = 300  # To match SpaCY vectors
SPACY_VECTOR_TYPE = np.float32




if False:
    text = "Blah blah. What's that?"
    tokens = tokenize(text, token_vector)
    print(tokens)
    print(sorted(token_vector))
    print([(v.shape, v.dtype) for v in token_vector.values()])
    assert False




# ## Building the model
#
# We'll start by creating a *bag of words* representation, as a *term document matrix*. We'll use
# ngrams, as suggested in the NBSVM paper.
if False:
    s = '''It turns out "why is" that's using. Doesn't it? Can't i'''
    t = tokenize(s)
    print(t)
    assert False




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
    r_1 = pr(1, x, y)
    r_0 = pr(0, x, y)
    r = r_1 - r_0
    x_nb = x * r
    print('** x, r, x_nb, y:', x.shape, r.shape, x_nb.shape, y.shape)
    m = LogisticRegression(C=4, dual=True)
    return m.fit(x_nb, y), r


class ClfGloveNBSpace:

    def __init__(self, label_cols, model_dir):
        self.label_cols = label_cols
        self.model_dir = model_dir
        self.m = {}
        self.r = {}
        self.text_tokens_path = os.path.join(self.model_dir, 'text.tokens.json')
        self.token_vector_path = os.path.join(self.model_dir, 'token.vector.pkl')
        self.text_tokens = load_json(self.text_tokens_path, {})
        self.token_vector = load_pickle(self.token_vector_path, {})
        self.text_tokens_len = len(self.text_tokens)
        self.token_vector_len = len(self.token_vector)
        self.nlp = spacy.load('en_core_web_lg')

    def _save(self):
        print('_save 1:', self.text_tokens_len, len(self.text_tokens), self.text_tokens_path)
        print('_save2:', self.token_vector_len, len(self.token_vector), self.token_vector_path)

        if self.text_tokens_len != len(self.text_tokens):
            save_json(self.text_tokens_path, self.text_tokens)
            self.text_tokens_len = len(self.text_tokens)
        if self.token_vector_len != len(self.token_vector):
            save_pickle(self.token_vector_path, self.token_vector)
            self.token_vector_len = len(self.token_vector)

    def tokenize(self, text):
        """Use SpaCy tokenization and word vectors"""
        doc = self.nlp(text)
        tokens = []
        for t in doc:
            tokens.append(t.text)
            self.token_vector[t.text] = t.vector
        return tokens

    def tokenize_df(self, df):
        """Return a list of token lists
            Each token list corresponds to a row of df
        """
        t0 = time.clock()
        tokens_list = []
        for text in df[COMMENT]:
            tokens = self.text_tokens.get(text)
            if not tokens:
                tokens = self.tokenize(text)
                self.text_tokens[text] = tokens
            tokens_list.append(tokens)
        print('tokens: %d %1.f sec %.2f sec / token' % (
            len(tokens),
            time.clock() - t0,
            (time.clock() - t0) / len(tokens)))
        self._save()
        return tokens_list

    def compute_ngram_vector(self, token_list, n):
        """Compute an embedding vector for all n-grams in token_list
        """
        vec = np.zeros((n, SPACY_VECTOR_SIZE), dtype=np.float64)
        n_vecs = len(token_list) - n + 1
        for i in range(n_vecs):
            for j in range(n):
                vec[j] += self.token_vector[token_list[i + j]]
        vec /= n_vecs
        return np.reshape(vec, n * SPACY_VECTOR_SIZE)

    def compute_ngram_matrix(self, token_matrix, n):
        print('@@', type(token_matrix),  type(token_matrix[0]))
        mat = np.zeros((len(token_matrix), n * SPACY_VECTOR_SIZE), dtype=np.float64)
        for i, token_list in enumerate(token_matrix):
            assert isinstance(token_list, list), (i, token_list)
            print(i, type(token_list), token_list)
            mat[i, :] = self.compute_ngram_vector(token_list, n)
        return mat

    def fit(self, train):
        t0 = time.clock()
        train_tokens = self.tokenize_df(train)
        print('train_tokens: %1.f sec %.2f sec / token' % (time.clock() - t0, (time.clock() - t0) / len(train_tokens)))
        x = self.compute_ngram_matrix(train_tokens, N_GRAM)
        for i, col in enumerate(self.label_cols):
            self.m[col], self.r[col] = fit_model(x, train[col])

    def predict(self, test):
        label_cols = self.label_cols
        preds = np.zeros((len(test), len(label_cols)))
        test_tokens = self.tokenize_df(test)
        test_x = self.compute_ngram_matrix(test_tokens, N_GRAM)
        for i, col in enumerate(self.label_cols):
            print('^' * 80)
            print('fit', i, col)
            m, r = self.m[col], self.r[col]
            x_nb = test_x * r
            preds[:, i] = m.predict_proba(x_nb)[:, 1]
        return preds

