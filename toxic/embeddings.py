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
from utils import DATA_ROOT, dim, xprint, load_pickle, save_pickle
from token_spacy import islowercase


PAD = '[PAD]'
OOV = '[OOV]'

EMBEDDINGS_DIR = 'embeddings'

GLOVE_SETS = {
    'twitter': ('glove.twitter.27B', tuple([25, 50, 100, 200])),
    '6B': ('glove.6B', tuple([50, 100, 200, 300])),
    '840B': ('glove.840B.300d', tuple([300]))
}

GLOVE_COMBOS = [(k, n) for k, v in GLOVE_SETS.items() for n in v[1]]
GLOVE_COMBOS.sort(key=lambda kn: (kn[1], kn[0]))


def local_path(path, ext):
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]
    return join(EMBEDDINGS_DIR, '%s.%s' % (name, ext))


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
embeddings_data = None


def get_embeddings_index(embed_name, embed_size):
    global embeddings_index, embeddings_data, embeddings_name, embedddings_size

    if embeddings_name != embed_name or embedddings_size != embed_size:
        if embeddings_index is not None:
            del embeddings_index
            embeddings_index = None
        embeddings_name = embed_name
        embedddings_size = embed_size

    if embeddings_index is None or embeddings_data is None:
        assert embed_name in GLOVE_SETS, embed_name
        embeddings_path = get_embedding_path(embed_name, embed_size)
        xprint('get_embeddings_index: embeddings_path=%s' % embeddings_path)

        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        embeddings_local = local_path(embeddings_path, 'pkl')
        embeddings_local_data = local_path(embeddings_path, 'npy')

        if not os.path.exists(embeddings_local):
            embeddings_index = {}
            embeddings_list = []
            with open(embeddings_path, 'rb') as f:
                t0 = time.clock()
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    embeddings_index[str(parts[0], 'utf-8')] = i
                    embeddings_list.append(np.asarray(parts[1:], dtype=np.float32))
                    if (i + 1) % 200000 == 0:
                        print('%7d embeddings %4.1f sec' % (i + 1, time.clock() - t0))

            print('%7d embeddings %4.1f sec' % (i + 1, time.clock() - t0))
            save_pickle(embeddings_local, embeddings_index)
            # save_pickle(embeddings_local + 'arr.pkl', embeddings_list)
            w = len(embeddings_list[0])
            data = np.empty((len(embeddings_list), w), dtype=np.float32)
            for i, x in enumerate(embeddings_list):
                data[i, :] = x
            np.save(embeddings_local_data, data, fix_imports=False)

            xprint('%7d embeddings %s %4.1f sec -> %r %r' % (
                len(embeddings_index), dim(data), time.clock() - t0,
                embeddings_local, embeddings_local_data))

        embeddings_index = load_pickle(embeddings_local)
        embeddings_data = np.load(embeddings_local_data)

    return embeddings_index, embeddings_data


def test_embeddings(embed_name, embed_size):
    ei, data = get_embeddings_index(embed_name, embed_size)
    assert ei
    print('got embeddings %s:%d embeddings=%d' % (embed_name, embed_size, len(ei)))
    keys = sorted(ei)
    keys2 = [keys[i * 1001 % len(keys)] for i in range(10)]
    keys3 = [keys[i * 10001 % len(keys)] for i in range(10)]
    keys4 = [keys[i * 100001 % len(keys)] for i in range(10)]
    print([embed_name, embed_size], len(ei), type(keys[0]), keys[:10], keys2, keys3, keys4)
    assert isinstance(keys[0], str)
    lowercase = all(islowercase(k) for k in ei)
    print('%s:%d lowercase=%s' % (embed_name, embed_size, lowercase))
    print(data[0, :10])
    return len(ei), lowercase


def get_embeddings(embed_name, embed_size, max_features, word_list, word_index):
    """Returns: embedding matrix n_words x embed_size
                n_words <= max_features
    """
    assert embed_name in GLOVE_SETS, embed_name
    embeddings_index, data = get_embeddings_index(embed_name, embed_size)

    # Use these vectors to create our embedding matrix, with random initialization for words
    # that aren't in GloVe. We'll use the same mean and stdev of embeddings the GloVe has when
    # generating the random init.
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    xprint('emb_mean=%.3f emb_std=%.3f' % (emb_mean, emb_std))

    for i, w in enumerate(word_list[2:]):
        assert islowercase(w), (i, w)

    vocab = [OOV, PAD] + [word for word in word_list if word in embeddings_index]
    vocab = vocab[:max_features]
    oov = [word for word in word_list if word not in embeddings_index]
    assert vocab, (len(word_list), len(word_index), len(embeddings_index),
         word_list[:20], sorted(embeddings_index)[:20])

    embeddings = np.random.normal(emb_mean, emb_std, (len(vocab), embed_size))
    # embeddings[0, :] = np.random.normal(emb_mean, emb_std, embed_size)   # OOV
    embeddings[1, :] = np.zeros(embed_size, dtype=np.float32)  # PAD
    for i, word in enumerate(vocab[2:]):
        embeddings[i:, :] = data[embeddings_index[word]]

    xprint('get_embeddings: oov=%d %s' % (len(oov), [(w, word_index[w]) for w in oov[:50]]))

    xprint('get_embeddings: embed_name=%s embed_size=%d'
           '\n max_features=%d word_list=%d word_index=%d'
           '\n -> embeddings_index=%d'
           '\n -> embeddings=%s vocab=%d=%.3f'
        % (embed_name, embed_size, max_features, len(word_list), len(word_index),
           len(embeddings_index), dim(embeddings), len(vocab), len(vocab) / len(word_list)))

    reduced_index = {w: i for i, w in enumerate(vocab)}
    xprint('get_embeddings: reduced_index= %d - %d (%d values)' % (
        min(reduced_index.values()),
        max(reduced_index.values()),
        len(reduced_index)))
    assert 0 <= min(reduced_index.values()) and max(reduced_index.values()) < embeddings.shape[0]
    del embeddings_index
    del data
    return embeddings, reduced_index


if __name__ == '__main__':
    if True:
        results = []
        for embed_name, embed_size in GLOVE_COMBOS:
            n, lowercase = test_embeddings(embed_name, embed_size)
            results.append((embed_name, embed_size, n, lowercase))
        for i, (embed_name, embed_size, n, lowercase) in enumerate(results):
            print('%4d: %10s:%-3d %7d %s' % (i, embed_name, embed_size, n, lowercase))

    if False:
        test_embeddings('840B', 300)
