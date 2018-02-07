# from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
from utils import orders_name, local_path, load_json
from classifier_bank import compute_score


def make_Xy():
    orders = pd.read_pickle(local_path)
    print('%s: %s' % (orders_name, list(orders.shape)))

    Xcols = [col for col in orders.columns if orders[col].dtype == np.int32]
    print(len(Xcols))
    orders = orders[Xcols]
    print('%s: %s' % (orders_name, list(orders.shape)))

    converted_quotes = load_json('converted_quotes.json')
    converted = {q for q, s in converted_quotes if s}

    quotes = orders[orders['isQuote'] == 1]
    print('%s: %s' % ('quotes', list(quotes.shape)))
    quotes = quotes[[col for col in quotes.columns if col not in
                     {'endCustomerId', 'originalEndCustomerId'}]]
    print('%s: %s' % ('quotes', list(quotes.shape)))

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


def find_best_features(X, y):
    for stat in chi2, f_classif, mutual_info_classif:
        print('-' * 80)
        for k in range(1, 10):
            clf = SelectKBest(stat, k=k).fit(X.values, y.converted.values)
            X_new = clf.transform(X)
            support = clf.get_support(indices=True)
            columns = [X.columns[i] for i in support]
            print(k, columns)


def show_scores(X, y):
    print('-' * 80)
    print('show_scores: X=%s y=%s columns=%s' % (list(X.shape), list(y.shape), list(X.columns)))

    classifier_score, metrics = compute_score(X, y)
    for i, met in metrics:
        print('-' * 80)
        print('%d %s' % (i, met))
        for j, name in enumerate(sorted(classifier_score, key=lambda k: -classifier_score[k][i])):
            print('%4d: %.3f %s' % (j, classifier_score[name][i], name), flush=True)


def show_scores_feature(X, y, stat, k):
    print('-' * 80)
    print('show_scores_feature: X=%s y=%s stat=%s k=%d' % (list(X.shape), list(y.shape),
        stat.__name__, k))

    clf = SelectKBest(stat, k=k).fit(X.values, y.converted.values)
    support = clf.get_support(indices=True)
    columns = [X.columns[i] for i in support]
    Xfeat = X[columns]
    show_scores(Xfeat, y)


def show_balance(y):
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print('%2d: %6d %4.1%%f' % (u, c, c / len(y)))
    # print(np.asarray((unique, counts)).T)


def resample(X, y, fraction=0.1):
    X_columns = X.columns
    y_columns = y.columns

    print('~' * 80)
    print('@@-\n', y.converted.value_counts())
    print('@@0 - Original')
    show_balance(y.values)

    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_sample(X, y)
    X, y = X_res, y_res
    print('@@1 - Oversampled')
    show_balance(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=fraction, random_state=42)
    X, y = X_test, y_test
    print('@@2 - Downasmpled')
    show_balance(y)

    X = pd.DataFrame(X, columns=X_columns)
    y = pd.DataFrame(y, columns=y_columns, index=X.index)
    print('@@+\n', y.converted.value_counts(), flush=True)

    return X, y


if __name__ == '__main__':
    # find_best_features(X, y)
    X, y = make_Xy()
    X, y = resample(X, y)
    show_scores_feature(X, y, chi2, k=4)
