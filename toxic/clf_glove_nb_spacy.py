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
import os
import spacy
import time
from utils import COMMENT, save_json, load_json, save_pickle, load_pickle


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
    r_1 = pr(1, x, y)
    r_0 = pr(0, x, y)
    r = r_1 - r_0
    x_nb = x * r
    print('** x, r, x_nb, y:', x.shape, r.shape, x_nb.shape, y.shape)
    m = LogisticRegression(C=4, dual=True)
    return m.fit(x_nb, y), r


class ClfGloveNBSpace:

    def __init__(self, label_cols, model_dir, n_gram):
        self.label_cols = label_cols
        self.model_dir = model_dir
        self.n_gram = n_gram
        self.m = {}
        self.r = {}
        self.text_tokens_path = os.path.join(self.model_dir, 'text.tokens.json')
        self.token_vector_path = os.path.join(self.model_dir, 'token.vector.pkl')
        self.text_tokens = load_json(self.text_tokens_path, {})
        self.token_vector = load_pickle(self.token_vector_path, {})
        self.text_tokens_len = len(self.text_tokens)
        self.token_vector_len = len(self.token_vector)
        self.nlp = spacy.load('en_core_web_lg')

    def _save(self, min_delta=0):
        if self.text_tokens_len + min_delta < len(self.text_tokens):
            print('_save 1: %7d = %7d + %4d %s' % (len(self.text_tokens),
                self.text_tokens_len, len(self.text_tokens) - self.text_tokens_len,
                self.text_tokens_path))
            save_json(self.text_tokens_path, self.text_tokens)
            self.text_tokens_len = len(self.text_tokens)
        if self.token_vector_len + 2 * min_delta < len(self.token_vector):
            print('_save 2: %7d = %7d + %4d %s' % (len(self.token_vector),
                self.token_vector_len, len(self.token_vector) - self.token_vector_len,
                self.token_vector_path))
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
        print('tokenize_df: %d %d' % (len(df), self.text_tokens_len))
        t0 = t1 = time.clock()
        tokens_list = []
        n_tokenized = 0
        for i, text in enumerate(df[COMMENT]):
            tokens = self.text_tokens.get(text)
            if not tokens:
                tokens = self.tokenize(text)
                self.text_tokens[text] = tokens
                n_tokenized += 1
            tokens_list.append(tokens)
            t = time.clock()
            if t > t1 + SAVE_TIME or i + 1 == len(df):
                print('tokenize_df: %d texts, %d new, %1.f sec (%.2f sec / token)' % (
                    i + 1, n_tokenized,
                    time.clock() - t0,
                    (time.clock() - t0) / max(n_tokenized, 1)))
                t1 = t
                self._save(min_delta=SAVE_ITEMS)
        self._save()
        return tokens_list

    def compute_ngram_vector(self, token_list, n):
        """Compute an embedding vector for all n-grams in token_list
        """
        vec = np.zeros((n, SPACY_VECTOR_SIZE), dtype=np.float64)
        n_toks = len(token_list)
        for i in range(n_toks):
            for j in range(min(n, n_toks - i)):
                vec[j] += self.token_vector[token_list[i + j]]
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
        for i, col in enumerate(self.label_cols):
            self.m[col], self.r[col] = fit_model(x, train[col])

    def predict(self, test):
        label_cols = self.label_cols
        preds = np.zeros((len(test), len(label_cols)))
        test_tokens = self.tokenize_df(test)
        test_x = self.compute_ngram_matrix(test_tokens)
        for i, col in enumerate(self.label_cols):
            print('fit', i, col)
            m, r = self.m[col], self.r[col]
            x_nb = test_x * r
            preds[:, i] = m.predict_proba(x_nb)[:, 1]
        return preds
