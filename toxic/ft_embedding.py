import numpy as np
from collections import defaultdict
from keras.preprocessing import text, sequence
import os
from utils import dim, xprint
from gru_framework import X_train0, X_test0


HACK = False
np.random.seed(42)

os.environ['OMP_NUM_THREADS'] = '4'

EMBEDDING_FILE = '~/data/models/fasttext/crawl-300d-2M.vec'
EMBEDDING_FILE = os.path.expanduser(EMBEDDING_FILE)
EMBEDDING_FILE = os.path.expanduser(EMBEDDING_FILE)
assert os.path.exists(EMBEDDING_FILE)


embed_size = 300
char_embed_size = 25

char_index_map = {}


def get_char_index(max_features, char_maxlen):
    global char_index_map
    if char_maxlen not in char_index_map:
        char_count = defaultdict(int)
        for doc in X_train0:
            for c in doc[:char_maxlen]:
                char_count[c] += 1
        char_list = sorted(char_count, key=lambda c: (-char_count[c], c))
        char_list = ['UNK'] + char_list
        char_list = char_list[:max_features]
        char_index_map[char_maxlen] = {c: i for i, c in enumerate(char_list)}
    print('*** char_index=%d' % len(char_index_map[char_maxlen]))
    return char_index_map[char_maxlen]


tokenizer_map = {}


def get_tokenizer(max_features, maxlen):
    global tokenizer_map
    if (max_features, maxlen) not in tokenizer_map:
        tokenizer = text.Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(list(X_train0) + list(X_test0))
        tokenizer_map[(max_features, maxlen)] = tokenizer
    return tokenizer_map[(max_features, maxlen)]


def tokenize(max_features, maxlen, X):
    tokenizer = get_tokenizer(max_features, maxlen)
    X = tokenizer.texts_to_sequences(X)
    X = sequence.pad_sequences(X, maxlen=maxlen)
    return X


def char_tokenize(max_features, maxlen, char_maxlen, X):
    char_index = get_char_index(max_features, maxlen)
    unk_idx = char_index['UNK']
    X_char = np.zeros((X.shape[0], char_maxlen), dtype=np.int32)
    for i in range(X.shape[0]):
        for j, c in enumerate(X[i][:char_maxlen]):
            X_char[i, j] = char_index.get(c, unk_idx)
    return X_char


def tokenize_all(max_features, maxlen):
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X_train0) + list(X_test0))
    X_train = tokenizer.texts_to_sequences(X_train0)
    X_test = tokenizer.texts_to_sequences(X_test0)
    x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    return tokenizer.word_index, x_train, x_test


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


embeddings_index = None


def get_embeddings(max_features, maxlen):
    global embeddings_index

    print('get_embeddings: embeddings_index=%s' % (embeddings_index is not None))
    if HACK:
        print('HACKING !!!!!!!!!!!!!!!!!!!!!')

    if embeddings_index is None:
        embeddings_index = {}
        with open(EMBEDDING_FILE) as f:
            for i, o in enumerate(f):
                w, vec = get_coefs(*o.rstrip().rsplit(' '))
                embeddings_index[w] = vec
                if HACK:
                    if i > 11000:  # !@#$
                        break
        # embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

    tokenizer = get_tokenizer(max_features, maxlen)
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    xprint('embedding_matrix=%s' % dim(embedding_matrix))
    return embedding_matrix


char_embeddings_index = None


def get_char_embeddings(max_features, maxlen):
    global char_embeddings_index

    print('get_char_embeddings: char_embeddings_index=%s' % (char_embeddings_index is not None))

    char_index = get_char_index(max_features, maxlen)
    n_chars = min(max_features, len(char_index))
    print('get_char_embeddings: n_chars=%d max_features=%d char_index=%d' % (
        n_chars, max_features, len(char_index)))
    # embedding_matrix = np.zeros((n_chars, char_embed_size))
    embedding_matrix = np.random.normal(0.0, 1.0, (n_chars, char_embed_size))
    embedding_matrix = np.clip(embedding_matrix, -1.0, 1.0)
    xprint('embedding_matrix=%s' % dim(embedding_matrix))
    return embedding_matrix
