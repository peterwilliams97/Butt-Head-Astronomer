# -*- coding: utf-8 -*-
"""
    Based on https://www.dataquest.io/blog/kaggle-tutorial/
"""
from __future__ import division, print_function
# import sys
import os
# import re
# import numpy as np
import pandas as pd
# from pandas import DataFrame, Series
# from glob import glob
from pprint import pprint
# from sklearn import cross_validation, utils
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from time import time
import random
import ml_metrics as metrics
from sklearn.decomposition import PCA


def f5(test, predictions):
    target = [[l] for l in test['hotel_cluster']]
    return metrics.mapk(target, predictions, k=5)


if False:
    train = pd.read_csv(os.path.join('data', 'train.csv')
        # , nrows=10000
        )
    cluster_values = train['hotel_cluster'].value_counts()
    print(cluster_values)
    print('%d values' % len(cluster_values))
    print('%d total' % cluster_values.sum())
    fractions = cluster_values / cluster_values.sum()
    print(fractions.describe())

if True:
    for name in 'train.csv', 'test.csv', 'destinations.csv':
        print('-' * 80)
        print(name)
        t0 = time()
        df = pd.read_csv(os.path.join('data', name), nrows=10)
        print('%.f secs' % (time() - t0))
        print(df.shape)
        pprint(list(enumerate([(col, df[col].dtype) for col in df.columns])))
        for idx, row in df.iterrows():
            print(idx, row)
    assert False

if False:
    train = pd.read_csv(os.path.join('data', 'train.csv'))
    test = pd.read_csv(os.path.join('data', 'test.csv'))

if False:
    test_ids = set(test['user_id'].unique())
    train_ids = set(train['user_id'].unique())
    common_ids = test_ids & train_ids
    intersection_count = len(test_ids & train_ids)
    print('train_ids:', len(train_ids))
    print('test_ids:', len(test_ids))
    print('test_ids & train_ids:', len(test_ids & train_ids))
    assert intersection_count == len(test_ids), (intersection_count, len(test_ids))

    train['date_time'] = pd.to_datetime(train['date_time'])
    train['year'] = train['date_time'].dt.year
    train['month'] = train['date_time'].dt.month


def synthesize_year_month(df):
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['year'] = df['date_time'].dt.year
    df['month'] = df['date_time'].dt.month


if False:
    # Sample training data
    unique_users = train.user_id.unique()

    sel_user_ids = [unique_users[i] for i in sorted(random.sample(range(len(unique_users)), 10000))]
    sel_train = train[train.user_id.isin(sel_user_ids)]

    sel_train.to_csv(os.path.join('derived', 'train10000.csv'))

sel_train = pd.read_csv(os.path.join('derived', 'train10000.csv'))
synthesize_year_month(sel_train)
t1 = sel_train[((sel_train.year == 2013) | ((sel_train.year == 2014) & (sel_train.month < 8)))]
t2 = sel_train[((sel_train.year == 2014) & (sel_train.month >= 8))]
print('sel_train', sel_train.shape)
print('t1       ', t1.shape)
print('t2       ', t2.shape)

t2 = t2[t2.is_booking == True]
print('t2       ', t2.shape)


def top_clusters(df, n=5):
    return list(df['hotel_cluster'].value_counts().head(n=n).index)

most_common_clusters = top_clusters(t1)
predictions = [most_common_clusters for i in range(t2.shape[0])]
result = f5(t2, predictions)
print('result=%g' % result)

cluster_corr = sel_train.corr()['hotel_cluster']
print(cluster_corr)
