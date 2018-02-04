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
from os.path import expanduser, join
import matplotlib.pyplot as plt
import spacy
import time
from utils import save_json, load_json, save_pickle, load_pickle


N_GRAM = 3
N_SAMPLES = -1  # > 0 for testing

data_dir = expanduser('/Users/pcadmin/data/toxic/')
train = pd.read_csv(join(data_dir, 'train.csv'))
test = pd.read_csv(join(data_dir, 'test.csv'))
subm = pd.read_csv(join(data_dir, 'sample_submission.csv'))
print('train,test,subm:', train.shape, test.shape, subm.shape)

if N_SAMPLES > 0:
    train = train[:N_SAMPLES]
    test = test[:N_SAMPLES]


nlp = spacy.load('en_core_web_lg')
SPACY_VECTOR_SIZE = 300  # To match SpaCY vectors
SPACY_VECTOR_TYPE = np.float32
token_vector = {}


def tokenize(s, token_vector):
    """Use SpaCy tokenization and word vectors"""
    doc = nlp(s)
    tokens = []
    for t in doc:
        tokens.append(t.text)
        token_vector[t.text] = t.vector
    return tokens


if False:
    text = "Blah blah. What's that?"
    tokens = tokenize(text, token_vector)
    print(tokens)
    print(sorted(token_vector))
    print([(v.shape, v.dtype) for v in token_vector.values()])
    assert False


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
# print(label_cols)
# print(train.columns)
# print(sorted(set(train.columns) - set(label_cols)))

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
if False:
    s = '''It turns out "why is" that's using. Doesn't it? Can't i'''
    t = tokenize(s)
    print(t)
    assert False

print('Tokenization:')
t0 = time.clock()
train_tokens = [tokenize(s, token_vector) for s in train[COMMENT]]
print('train_tokens: %1.f sec %.2f sec / token' % (time.clock() - t0, (time.clock() - t0) / len(train_tokens)))
t0 = time.clock()
test_tokens = [tokenize(s, token_vector) for s in test[COMMENT]]
print('test_tokens: %1.f sec %.2f sec / token' % (time.clock() - t0, (time.clock() - t0) / len(test_tokens)))

save_pickle('token.vector.pkl', token_vector)
save_json('train.tokens.json', train_tokens)
save_json('test.tokens.json', test_tokens)

token_vector = load_pickle('token.vector.pkl')
train_tokens = load_json('train.tokens.json')
test_tokens = load_json('test.tokens.json')


def compute_ngram_vector(token_list, n):
    """Compute an embedding vector for all n-grams in token_list
    """
    vec = np.zeros((n, SPACY_VECTOR_SIZE), dtype=np.float64)
    n_vecs = len(token_list) - n + 1
    for i in range(n_vecs):
        for j in range(n):
            vec[j] += token_vector[token_list[i + j]]
    vec /= n_vecs
    return np.reshape(vec, n * SPACY_VECTOR_SIZE)


def compute_ngram_matrix(token_matrix, n):
    mat = np.zeros((len(token_matrix), n * SPACY_VECTOR_SIZE), dtype=np.float64)
    for i, token_list in enumerate(token_matrix):
        mat[i, :] = compute_ngram_vector(token_list, n)
    return mat


# It turns out that using TF-IDF gives even better priors than the binarized features used in the
# paper. I don't think this has been mentioned in any paper before, but it improves leaderboard
# score from 0.59 to 0.55.
n = train.shape[0]
trn_term_doc = compute_ngram_matrix(train_tokens, N_GRAM)
test_term_doc = compute_ngram_matrix(test_tokens, N_GRAM)
print('trn_term_doc :', trn_term_doc.shape, trn_term_doc.dtype)
print('test_term_doc:', test_term_doc.shape, test_term_doc.dtype)


# Here's the basic naive bayes feature equation:
def pr(y_i, y):
    """
        y_i:  0 or 1
        y: the y vector
        returns: y_pred[1, m] 1 x m vector (x is an n x m vector)
                 y_pred[j] = sum p[:,j] / sum(p[:,:]) where p = x[y == y_i]

    """
    x_i = x[y == y_i]
    return (x_i.sum(0) + 1) / (x_i.sum() + 1)


x = trn_term_doc
test_x = test_term_doc
# print(x)


def fit_model(y):
    """Fit a model for one dependent at a time
    """
    y = y.values
    r_1 = pr(1, y)
    r_0 = pr(0, y)
    r = r_1 - r_0
    x_nb = x * r
    print('** x, r, x_nb, y:', x.shape, r.shape, x_nb.shape, y.shape)
    m = LogisticRegression(C=4, dual=True)
    return m.fit(x_nb, y), r


preds = np.zeros((len(test), len(label_cols)))

for i, j in enumerate(label_cols):
    print('^' * 80)
    print('fit', j)
    m, r = fit_model(train[j])
    x_nb = test_x * r
    preds[:, i] = m.predict_proba(x_nb)[:, 1]

print()

# And finally, create the submission file.
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns=label_cols)], axis=1)
submission.to_csv('submission002 .csv', index=False)
