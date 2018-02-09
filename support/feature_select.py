# from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from utils import ORDERS_NAME, ORDERS_LOCAL_PATH, dim, load_json, save_pickle
from classifier_bank import metrics, compute_score, compute_all_scores


def load_enumerations():
    enumerations = load_json('enumerations.json')
    idx_to_key = {col: {i: k for i, k in enumerate(enumerations[col])} for col in enumerations}
    key_to_idx = {col: {k: i for i, k in enumerate(enumerations[col])} for col in enumerations}

    # for col in enumerations:
    #     idx_to_key[col] = {i: k for i, k in enumerate(enumerations[col])}
    #     key_to_idx[col] = {k: i for i, k in enumerate(enumerations[col])}

    return idx_to_key, key_to_idx


def make_Xy():
    orders = pd.read_pickle(ORDERS_LOCAL_PATH)
    print('%s: %s' % (ORDERS_NAME, list(orders.shape)))

    Xcols = [col for col in orders.columns if orders[col].dtype == np.int32]
    print(len(Xcols))
    orders = orders[Xcols]
    print('%s: %s' % (ORDERS_NAME, list(orders.shape)))

    converted_quotes = load_json('converted_quotes.json')
    converted = {q for q, s in converted_quotes if s}

    quotes = orders[orders['isQuote'] == 1]

    _, key_to_idx = load_enumerations()
    CHANGE = key_to_idx['type']['CHANGE']
    quotes = quotes[quotes['type'] == CHANGE]

    print('%s: %s' % ('quotes', dim(quotes)))
    quotes = quotes[[col for col in quotes.columns if col not in
                     {'endCustomerId', 'originalEndCustomerId'}]]
    print('%s: %s' % ('quotes', dim(quotes)))

    y = pd.DataFrame(index=quotes.index)
    y_values = [i in converted for i in quotes['id'].values]
    y['converted'] = pd.Series(y_values, index=quotes.index, dtype=np.int32)
    print('y', y.shape, y['converted'].dtype)

    X = quotes[[col for col in quotes.columns if col != 'isQuote']]
    X = X[[col for col in X.columns if col != 'id']]
    X = X[[col for col in X.columns if col != 'customerId']]
    print('X', X.shape)
    # print(X.describe())
    return X, y


def find_best_features(X, y, max_k=10):
    for stat in chi2, f_classif, mutual_info_classif:
        print('-' * 80)
        print(stat.__name__)
        for k in range(1, max_k + 1):
            clf = SelectKBest(stat, k=k).fit(X.values, y.converted.values)
            # X_new = clf.transform(X)
            support = clf.get_support(indices=True)
            columns = [X.columns[i] for i in support]
            print(k, columns)


def show_scores(X_train, y_train, X_test, y_test):
    print('-' * 80)
    print('show_scores: X_train=%s y_train=%s X_test=%s y_test=%s' % (
        dim(X_train), dim(y_train),
        dim(X_test), dim(y_test)))

    if False:
        classifier_score = compute_all_scores(X_train, y_train, X_test, y_test)
        for i, met in enumerate(metrics):
            print('-' * 80)
            print('%d %s' % (i, met))
            for j, name in enumerate(sorted(classifier_score, key=lambda k: -classifier_score[k][i])):
                print('%4d: %.3f %s' % (j, classifier_score[name][i], name), flush=True)
    if True:
        scores = []
        for n_neighbors in range(4, 5):
            score = compute_score(X_train, y_train, X_test, y_test, n_neighbors)
            scores.append(score)
        for i, score in enumerate(scores):
            print('%3d: %s' % (i, score))
        return scores[0]


def show_scores_feature(X_train, y_train, X_test, y_test, stat, k):
    print('-' * 80)
    print('show_scores_feature: X_train=%s y_train=%s X_test=%s y_test=%s stat=%s k=%d' % (
        list(X_train.shape), list(y_train.shape),
        list(X_test.shape), list(y_test.shape),
        stat.__name__, k))

    clf = SelectKBest(stat, k=k).fit(X_train.values, y_train.converted.values)
    support = clf.get_support(indices=True)
    columns = [X_train.columns[i] for i in support]
    X_train_feat = X_train[columns]
    X_test_feat = X_test[columns]

    ohe = OneHotEncoder()
    X_feat = pd.concat([X_train_feat, X_test_feat])
    ohe.fit(X_feat)
    X_train_feat = ohe.transform(X_train_feat)
    X_test_feat = ohe.transform(X_test_feat)

    print('OneHotEncoder: X_train_feat=%s X_train_feat=%s' % (
        list(X_train_feat.shape), list(X_test_feat.shape)))

    return list(X_train_feat.shape), show_scores(X_train_feat, y_train, X_test_feat, y_test), columns


def beam_search_feature(X_train, y_train, X_test, y_test, beam_size, max_items=-1, patience=5):
    print('-' * 80)
    print('beam_search_feature: X_train=%s y_train=%s X_test=%s y_test=%s' % (
        dim(X_train), dim(y_train), dim(X_test), dim(y_test)))

    def get_score(cols):
        # print('$$', type(cols), cols)
        X_train_feat = X_train[list(cols)]
        X_test_feat = X_test[list(cols)]
        ohe = OneHotEncoder()
        X_feat = pd.concat([X_train_feat, X_test_feat])
        ohe.fit(X_feat)
        X_train_feat = ohe.transform(X_train_feat)
        X_test_feat = ohe.transform(X_test_feat)
        # print('OneHotEncoder: X_train_feat=%s X_train_feat=%s' % (dim(X_train_feat), dim(X_test_feat)))
        return compute_score(X_train_feat, y_train, X_test_feat, y_test)

    # m = number of columns
    M = X_train.shape[1]
    columns = list(X_train.columns)

    m_scores = []
    beam = [tuple()]
    best_f1 = 0.0
    best_m = 0
    for m in range(1, M):
        candidate_scores = []
        history = set()
        for b in beam:
            for k in columns:
                if k in b:
                    continue
                candidate = b + tuple([k])
                fz = frozenset(candidate)
                if fz in history:
                    continue
                history.add(fz)
                score = get_score(candidate)
                candidate_scores.append((candidate, score))

        if not candidate_scores:
            break
        if m == 1:
            assert all(len(c) == 1 for c, _ in candidate_scores)
            columns = [c[0] for c, _ in candidate_scores]
            if max_items > 0:
                columns = columns[:max_items]
            assert columns

        candidate_scores.sort(key=lambda ks: (-ks[1][1], ks[0]))
        m_scores.append((m, candidate_scores))
        beam = tuple(k for k, _ in candidate_scores[:beam_size])
        if candidate_scores[0][1][1] > best_f1:
            best_f1 = candidate_scores[0][1][1]
            best_m = m
        if patience >= 0 and m > best_m + patience:
            print('Patience! m=%d best_m=%d patience=%d' % (m, best_m, patience))
            break
        print('^^^ %d %s' % (m, beam), flush=True)
    return m_scores


def show_balance(y):
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print('%2d: %6d %4.1f%%' % (u, c, 100.0 * c / len(y)))
    # print(np.asarray((unique, counts)).T)


def resample(X, y, sample_fraction=0.1, test_size=0.3):
    X_columns = X.columns
    y_columns = y.columns
    n = len(X_columns)

    print('~' * 80)
    print('@@-\n', y.converted.value_counts())
    print('@@0 - Original')
    show_balance(y.values)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print('@@2 - y_train')
    show_balance(y_train)
    print('@@2 -  y_test')
    show_balance(y_test)
    assert X_train.shape[1] == n and X_test.shape[1] == n

    ros = RandomOverSampler(random_state=42)
    X_train, y_train = ros.fit_sample(X_train, y_train)
    X_test, y_test = ros.fit_sample(X_test, y_test)
    print('@@3 - Oversampled y_train')
    show_balance(y_train)
    print('@@3 - Oversampled y_test')
    show_balance(y_test)
    assert X_train.shape[1] == n and X_test.shape[1] == n

    if sample_fraction < 1.0:
        _, X_train, _, y_train = train_test_split(X_train, y_train, test_size=sample_fraction, random_state=43)
        _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=sample_fraction, random_state=44)
        print('@@2 - Downsampled y_train')
        show_balance(y_train)
        print('@@2 - Downsampled y_test')
        show_balance(y_test)
        assert len(X_train.shape) == 2 and len(X_test.shape) == 2, (X_train.shape, X_test.shape)
        assert X_train.shape[1] == n and X_test.shape[1] == n, (X_train.shape, X_test.shape)

    print('X_columns=%d %s' % (len(X_columns), X_columns))
    print('y_columns=%d %s' % (len(y_columns), y_columns))
    print('X_train=%-10s y_train=%s' % (list(X_train.shape), list(y_train.shape)))
    print('X_test =%-10s y_test =%s' % (list(X_test.shape), list(y_test.shape)))
    assert X_train.shape[1] == n and X_test.shape[1] == n

    X_train = pd.DataFrame(X_train, columns=X_columns)
    y_train = pd.DataFrame(y_train, columns=y_columns, index=X_train.index)
    X_test = pd.DataFrame(X_test, columns=X_columns)
    y_test = pd.DataFrame(y_test, columns=y_columns, index=X_test.index)
    print('@@+ y_train\n', y_train.converted.value_counts(), flush=True)
    print('@@+ y_test\n', y_test.converted.value_counts(), flush=True)

    return (X_train, y_train), (X_test, y_test)


def value_str(y):
    # 0    13909
    # 1     2357
    vals = y.value_counts()
    assert len(vals) <= 2, len(vals)
    v_0 = 0
    v_1 = 0
    if 0 in vals.index:
        v_0 = vals.loc[0]
    if 1 in vals.index:
        v_1 = vals.loc[1]
    n = len(y)
    assert v_0 + v_1 == n, (v_0, v_1, n)
    # r_0 = 100.0 * v_0 / n
    r_1 = 100.0 * v_1 / n
    return '%6d - %5d  (%4.1f%%)' % (v_0, v_1, r_1)


def show_splits(X, y, columns):

    idx_to_key, _ = load_enumerations()

    y = y['converted']
    print('+' * 80)
    print('%3s %-20s %s' % ('', 'all', value_str(y)))
    for col in columns:
        print('--', col)
        vals = X[col].value_counts()
        for i in vals.index:
            k = idx_to_key[col][i]
            idx = X[col] == i
            y_i = y[idx]
            print('%3d %-20s %s' % (i, k, value_str(y_i)))


if __name__ == '__main__':
    X, y = make_Xy()
    if False:
        find_best_features(X, y)
    if False:
        (X_train, y_train), (X_test, y_test) = resample(X, y, sample_fraction=1.0)
        results = []
        for k in range(1, 11):
            # chi2                10: [24478, 3527]  [0.8236972232788132, 0.8460899883778848, 0.8236972232788131]
            # f_classif           10: [24478, 1631]  [0.8236021300874857, 0.8453522300958731, 0.8236021300874857]
            # f_classif          10: [24478, 1631]  [0.8225561049828832, 0.8473244968090329, 0.8225561049828833]  XGBoost
            # mutual_info_classif 10: [24478, 3533]  [0.8235070368961582, 0.8458983726336766, 0.8235070368961582]
            shape, score, columns = show_scores_feature(X_train, y_train, X_test, y_test, chi2, k=k)
            results.append((k, shape, score, columns))
        print('#' * 80)
        for k, shape, score, columns in results:
            print('%4d: %-20s %s %s ' % (k, shape, score, columns))
        last_f1 = 0.0
        last_col = set()
        print('`' * 80)
        for k, shape, score, columns in results:
            f1 = score[1]
            col = set(columns)
            d_f1 = f1 - last_f1
            d_col = list(col - last_col)
            print('%4d: %-20s %g' % (k, d_col, d_f1))
            last_f1 = f1
            last_col = col
    if True:
        # X = X[X.columns[:5]]
        beam_size = 3
        max_items = -1
        (X_train, y_train), (X_test, y_test) = resample(X, y, sample_fraction=1.0)
        m_scores = beam_search_feature(X_train, y_train, X_test, y_test, beam_size=3, max_items=-1)
        save_pickle('m_scores.pkl', m_scores)

        print('*' * 80)
        print('beam_size=%d, max_items=%d' % (beam_size, max_items))
        last_f1 = 0
        for m, cols_scores in m_scores:
            f1 = cols_scores[0][1][1]
            print('%3d: f1=%.3f improvement=%+.3f' % (m, f1, f1 - last_f1))
            last_f1 = f1
            for i, (c, s) in enumerate(cols_scores[:3]):
                print('%5d: %s %s' % (i, s, c))
    if False:
        columns = ['Parent Region', 'Reseller Tier', 'resellerDiscountPercentage', 'type']
        show_splits(X, y, columns)


BEAM = """
  1: f1=0.826 improvement=+0.826
    0: [0.7898440471662229, 0.8260937991816178, 0.7898440471662229] ('Customer Country',)
    1: [0.6237162419170788, 0.7017411622823547, 0.6237162419170788] ('address1',)
    2: [0.6201027006466338, 0.6967741935483871, 0.6201027006466338] ('organizationName',)
  2: f1=0.847 improvement=+0.021
    0: [0.8222708254089007, 0.8473911978443701, 0.8222708254089007] ('Customer Country', 'address1')
    1: [0.8224610117915557, 0.8468039714449823, 0.8224610117915558] ('Customer Country', 'Reseller Name')
    2: [0.8214149866869532, 0.846643802057815, 0.8214149866869533] ('Customer Country', 'lastName')
  3: f1=0.848 improvement=+0.000
"""

OLD = """
    chi2 k=1 [0.8970740548938374, 0.9027332566618542, 0.8970740548938374]
    chi2 k=2 [0.8961677887105127, 0.9014742014742014, 0.8961677887105127]
    chi2 k=3 [0.8970999482133609, 0.902272280149518,  0.897099948213361]
    chi2 k=4 [0.8970740548938374, 0.9022500922168941, 0.8970740548938374]      =[90312, 2777]
    f_classif k=1  [0.8689280165717245, 0.87946471092485,   0.8689280165717245] =[90312,    4]
    f_classif k=2  [0.8663904712584153, 0.8791116109080687, 0.8663904712584154] =[90312,   16]
    f_classif k=3  [0.9035732780942517, 0.9095721431693458, 0.9035732780942517] =[90312,   32]
    f_classif k=4  [0.9616261004660798, 0.9619570797823185, 0.9616261004660797] =[90312,   38]
    f_classif k=5  [0.9629466597617815, 0.9633537350508335, 0.9629466597617815] =[90312,  688]
    f_classif k=10 [0.9612118073537027, 0.9617075664621677, 0.9612118073537027] =[90312, 3922]
    chi2 k=2
    chi2 k=2

    1 ['Reseller Tier']
    2 ['Parent Region', 'Reseller Tier']
    3 ['Parent Region', 'Reseller Tier', 'resellerDiscountPercentage']
    4 ['Parent Region', 'Reseller Tier', 'resellerDiscountPercentage', 'type']
    5 ['Parent Region', 'Reseller Tier', 'Customer Country', 'resellerDiscountPercentage', 'type']
    6 ['Parent Region', 'Reseller Tier', 'Customer Country', 'firstName', 'resellerDiscountPercentage', 'type']
    7 ['Parent Region', 'Reseller Tier', 'Reseller Code', 'Customer Country', 'firstName', 'resellerDiscountPercentage', 'type']
    8 ['Parent Region', 'Reseller Tier', 'Reseller Code', 'Customer Country', 'firstName', 'resellerCode', 'resellerDiscountPercentage', 'type']
    9 ['Parent Region', 'Reseller Tier', 'Reseller Code', 'Reseller Name', 'Customer Country', 'firstName', 'resellerCode', 'resellerDiscountPercentage', 'type']
"""

NEW = """
    --------------------------------------------------------------------------------
    stat.__name__
    1 ['Reseller Code']
    2 ['Reseller Code', 'resellerCode']
    3 ['Reseller Code', 'organizationName', 'resellerCode']
    4 ['Reseller Code', 'address1', 'organizationName', 'resellerCode']
    5 ['Reseller Code', 'address1', 'lastName', 'organizationName', 'resellerCode']
    6 ['Reseller Code', 'Customer Country', 'address1', 'lastName', 'organizationName', 'resellerCode']
    7 ['Reseller Code', 'Reseller Name', 'Customer Country', 'address1', 'lastName', 'organizationName', 'resellerCode']
    8 ['Reseller Code', 'Reseller Name', 'Customer Country', 'address1', 'firstName', 'lastName', 'organizationName', 'resellerCode']
    9 ['Reseller Code', 'Reseller Name', 'Reseller City', 'Customer Country', 'address1', 'firstName', 'lastName', 'organizationName', 'resellerCode']
    10 ['Reseller Code', 'Reseller Name', 'Reseller City', 'Customer Country', 'address1', 'city', 'firstName', 'lastName', 'organizationName', 'resellerCode']
    --------------------------------------------------------------------------------
    stat.__name__
    1 ['Super Region']
    2 ['Super Region', 'resellerDiscountPercentage']
    3 ['Super Region', 'Reseller Tier', 'resellerDiscountPercentage']
    4 ['Super Region', 'Reseller Tier', 'Customer Country', 'resellerDiscountPercentage']
    5 ['Super Region', 'Reseller Tier', 'Customer Country', 'currency', 'resellerDiscountPercentage']
    6 ['Super Region', 'Parent Region', 'Reseller Tier', 'Customer Country', 'currency', 'resellerDiscountPercentage']
    7 ['Super Region', 'Parent Region', 'Reseller Tier', 'Customer Country', 'firstName', 'currency', 'resellerDiscountPercentage']
    8 ['Super Region', 'Parent Region', 'Reseller Tier', 'Reseller Code', 'Customer Country', 'firstName', 'currency', 'resellerDiscountPercentage']
    9 ['Super Region', 'Parent Region', 'Reseller Tier', 'Reseller Code', 'Customer Country', 'firstName', 'currency', 'resellerCode', 'resellerDiscountPercentage']
    10 ['Super Region', 'Parent Region', 'Reseller Tier', 'Reseller Code', 'Reseller Name', 'Customer Country', 'firstName', 'currency', 'resellerCode', 'resellerDiscountPercentage']
    --------------------------------------------------------------------------------
    stat.__name__
    1 ['Customer Country']
    2 ['Customer Country', 'address1']
    3 ['Customer Country', 'address1', 'organizationName']
    4 ['Reseller Code', 'Customer Country', 'address1', 'organizationName']
    5 ['Customer Country', 'address1', 'lastName', 'organizationName', 'resellerCode']
    6 ['Reseller Code', 'Customer Country', 'address1', 'lastName', 'organizationName', 'resellerCode']
    7 ['Reseller Code', 'Customer Country', 'address1', 'city', 'firstName', 'lastName', 'organizationName']
    8 ['Reseller Code', 'Reseller Name', 'Customer Country', 'address1', 'firstName', 'lastName', 'organizationName', 'resellerCode']
    9 ['Reseller Code', 'Reseller Name', 'Customer Country', 'address1', 'city', 'firstName', 'lastName', 'organizationName', 'resellerCode']
    10 ['Reseller Code', 'Reseller Name', 'Customer Country', 'address1', 'city', 'zip', 'firstName', 'lastName', 'organizationName', 'resellerCode']


"""
