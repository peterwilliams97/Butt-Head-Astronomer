# coding: utf-8
"""
  LSTM + GloVe + Cross-validation + LearningRate changes + ...

    Notes
    1) GRU is very similar to LSTM and not better

    2) GloVe dimension is very important. I recommend to use GloVe 840b 300d if you can (it's very
    hard to use it in kaggle kernels)

    3) Cross Validation is interesting for hiperparameters tuning, but for higher score you shoudn't
     use validation_split

    4) First Epoch is very unstable. So I use small LR on first step

    5) Dataset size is small. So you may use some additional datasets and then finetune model

    6) It's hard not to overfit the model and I haven't found yet a good way to solve this problem.
    BatchNormalization/Dropout don't really help.
    https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/46494263247
    Maybe Reccurent Batch Normalization can help, but it is'not implemented in keras.

    7) Use Attention layer from here (AttLayer):
    https://github.com/dem-esgal/textClassifier/blob/master/textClassifierHATT.py

    Thanks to (https://www.kaggle.com/CVxTz/keras-bidirectional-lstm-baseline-lb-0-051)
"""
import os
import time
from os.path import join
import numpy as np
from utils import DATA_ROOT, dim, xprint


GLOVE_SETS = {
    'twitter': ('glove.twitter.27B', tuple([25, 50, 100, 200])),
    '6B': ('glove.6B', tuple([50, 100, 200, 300])),
    '840B': ('glove.840B.300d', tuple([300]))
}


def valid_embedding(embed_name, embed_size):
    _, glove_sizes = GLOVE_SETS[embed_name]
    return embed_size in glove_sizes


def get_embedding_path(embed_name, embed_size):
    glove_name, glove_sizes = GLOVE_SETS[embed_name]
    assert embed_size in glove_sizes, (embed_name, embed_size, glove_sizes)
    embedding_dir = join(DATA_ROOT, glove_name)
    assert os.path.exists(embedding_dir), embedding_dir
    if embed_name == '840B':
        embedding_path = join(embedding_dir, '%s.txt' % glove_name)
    else:
        embedding_path = join(embedding_dir, '%s.%dd.txt' % (glove_name, embed_size))
    assert os.path.exists(embedding_path), embedding_path
    return embedding_path


if False:
    for embed_name, (glove_name, glove_sizes) in GLOVE_SETS.items():
        for embed_size in glove_sizes:
            embeddings_path = get_embedding_path(embed_name, embed_size)
            assert os.path.exists(embeddings_path), embeddings_path


embeddings_index = None
embeddings_name = None
embedddings_size = None


def get_embeddings_index(embed_name, embed_size):
    global embeddings_index, embeddings_name, embedddings_size

    if embeddings_name != embed_name or embedddings_size != embed_size:
        if embeddings_index is not None:
            del embeddings_index
            embeddings_index = None
        embeddings_name = embed_name
        embedddings_size = embed_size

    if embeddings_index is None:
        assert embed_name in GLOVE_SETS, embed_name
        embeddings_path = get_embedding_path(embed_name, embed_size)
        xprint('get_embeddings_index: embeddings_path=%s' % embeddings_path)
        embeddings_index = {}
        with open(embeddings_path, 'rb') as f:
            t0 = time.clock()
            for i, line in enumerate(f):
                parts = line.strip().split()
                embeddings_index[parts[0]] = np.asarray(parts[1:], dtype='float32')
                if (i + 1) % 200000 == 0:
                    print('%7d embeddings %4.1f sec' % (i + 1, time.clock() - t0))
        xprint('%7d embeddings %4.1f sec' % (len(embeddings_index), time.clock() - t0))
    return embeddings_index


if False:
    for embed_name, (glove_name, glove_sizes) in GLOVE_SETS.items():
        for embed_size in glove_sizes:
            ei = get_embeddings_index(embed_name, embed_size)
            assert ei


def get_embeddings(embed_name, embed_size, max_features, word_list, word_index):
    """Returns: embedding matrix n_words x embed_size
                n_words <= max_features
    """
    assert embed_name in GLOVE_SETS, embed_name
    embeddings_index = get_embeddings_index(embed_name, embed_size)

    # Use these vectors to create our embedding matrix, with random initialization for words
    # that aren't in GloVe. We'll use the same mean and stdev of embeddings the GloVe has when
    # generating the random init.
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    xprint('emb_mean=%.3f emb_std=%.3f' % (emb_mean, emb_std))

    embeddings = np.empty((len(word_list), embed_size), dtype=np.float32)
    embeddings[0, :] = np.random.normal(emb_mean, emb_std, embed_size)   # OOV
    embeddings[1, :] = np.zeros(embed_size, dtype=np.float32)  # PAD
    for i, word in enumerate(word_list[2:]):
        embeddings[i:, :] = embeddings_index.get(word)

    xprint('get_embeddings: embed_name, embed_size, max_features, word_list, word_index -> embeddings=%s'
        % (embed_name, embed_size, max_features, len(word_list), len(word_index), dim(embeddings)))

    return embeddings

