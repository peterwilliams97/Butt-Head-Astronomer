# coding: utf-8
"""
    Another Keras solution to Kaggle Toxic Comment challenge
"""
from utils import xprint_init, xprint
from framework import evaluate, seed_random, auc_score
from clf_lstm_glove import ClfLstmGlove, valid_embedding


epochs = 1
submission_name = 'lstm_glove_explore'


def valid_embedding_params(n_hidden, dropout, max_features, learning_rate, maxlen, n_folds, embed_name,
    embed_size):
    print('~~~', embed_name, embed_size, valid_embedding(embed_name, embed_size))
    return valid_embedding(embed_name, embed_size)


def evaluate_params(i, n_hidden, dropout, max_features, learning_rate, maxlen, n_folds, embed_name,
    embed_size, n=1):

    def get_clf():
        return ClfLstmGlove(n_hidden=n_hidden, embed_name=embed_name, embed_size=embed_size, maxlen=maxlen,
            max_features=max_features, dropout=dropout, epochs=epochs, learning_rate=learning_rate,
            n_folds=1)

    xprint('#' * 80)
    xprint(get_clf())
    seed_random(seed=i + 1000)

    xprint('evaluate_params(n_hidden=%d, dropout=%.3f, max_features=%d, learning_rate=%s' % (
        n_hidden, dropout, max_features, learning_rate))
    xprint(get_clf())

    auc = evaluate(get_clf, n=n)
    xprint('=' * 80)
    return auc, str(get_clf())


def get_auc(i, params, n):
    n_hidden, dropout, max_features, learning_rate, maxlen, n_folds, embed_name, embed_size = params
    print('$$', n_hidden, dropout, max_features, learning_rate, maxlen, n_folds, embed_name, embed_size)
    assert isinstance(max_features, int), max_features
    return evaluate_params(i, n_hidden, dropout, max_features, learning_rate, maxlen, n_folds,
        embed_name, embed_size, n=n)


def blend(bval, k, kval):
    p0 = list(bval)[:k] if k > 0 else []
    p1 = [l[0] for l in list_list[len(p0) + 1:]]
    params = tuple(p0 + [kval] + p1)
    print('****', ((len(params), len(list_list)), k, params, (p0, kval, p1)))
    assert len(params) == len(list_list), ((len(params), len(list_list)),
        k, params, (p0, kval, p1))
    for v1, v2 in zip(bval, params):
        assert type(v1) == type(v2)
    return params


def beam_search(list_list, beam_size=3, n=1):
    xprint('-' * 80)
    xprint('beam_search:')

    scores = []
    beam = [tuple()]
    params_auc = {}

    i = 0
    for k, klist in enumerate(list_list):
        for bval in beam:
            for kval in klist:
                params = blend(bval, k, kval)
                if params in params_auc:
                    continue
                if not valid_embedding_params(*params):
                    continue
                print('###', len(params), params)
                try:
                    auc, desc = get_auc(i, params, n)
                except Exception as e:
                    print('&&&', e)
                    continue
                score, col_scores = auc_score(auc)
                scores.append((score, col_scores, params, desc))
                params_auc[params] = col_scores
                i += 1
        scores.sort(key=lambda x: (-x[0], x[1:]))
        beam = [params for _, _, params, _ in scores[:beam_size]]
        xprint('!' * 80)
        with open('all.results.txt', 'wt') as f:
            for i, (score, col_scores, params, desc) in enumerate(scores):
                if i < 10:
                    xprint('%4d: auc=%.3f %s %s %s' % (i, score, col_scores, params, desc))
                print('%4d: auc=%.3f %s %s %s' % (i, score, col_scores, params, desc), file=f)
        xprint(k, '|' * 80)


# n_hidden, dropout, max_features, learning_rate
n_hidden_list = [50, 100, 200, 75]
dropout_list = [0.1, 0.2, 0.15, 0.3]
max_features_list = [20000, 30000]  # , 40000] # 80000]
learning_rate_list = [
    (0.002, 0.003, 0.000),
    (0.002, 0.002, 0.002, 0.003, 0.000),
    (0.002, 0.003, 0.003, 0.003, 0.000),
    (0.002, 0.003, 0.000, 0.001)
]
maxlen_list = [100, 150, 70, 200]
n_folds_list = [1, 2, 3]
embed_name_list = ['6B', '840B', 'twitter']
embed_size_list = [50, 300, 200, 100, 25]

list_list = [
    n_hidden_list,
    dropout_list,
    max_features_list,
    learning_rate_list,
    maxlen_list,
    n_folds_list,
    embed_name_list,
    embed_size_list,
]

xprint_init(submission_name)
beam_search(list_list, beam_size=3, n=1)
xprint('$' * 80)
