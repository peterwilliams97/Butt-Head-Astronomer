# coding: utf-8
"""
"""
import json
import pickle
import os
import numpy as np
import pandas as pd


data_dir = os.path.expanduser('~/data/support-predictions/')
ORDERS_NAME = 'quote-order'
ORDERS_LOCAL_PATH = '%s.pkl' % ORDERS_NAME


def dim(x):
    return list(x.shape)


def load_json(path, default=None):
    if default is not None and not os.path.exists(path):
        return default
    try:
        with open(path, 'r') as f:
            obj = json.load(f)
    except json.decoder.JSONDecodeError:
        print('load_json failed: path=%r' % path)
        raise
    return obj


temp_json = 'temp.json'


def save_json(path, obj):
    with open(temp_json, 'w') as f:
        json.dump(obj, f, indent=4, sort_keys=True)
    # remove(path)
    os.renames(temp_json, path)


def load_pickle(path, default=None):
    if default is not None and not os.path.exists(path):
        return default
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
    except:
        print('load_pickle failed: path=%r' % path)
        raise
    return obj


temp_pickle = 'temp.pkl'


def save_pickle(path, obj):
    with open(temp_pickle, 'wb') as f:
        pickle.dump(obj, f)
    os.renames(temp_pickle, path)


def convert_categoricals(df, enumerations):
    df2 = pd.DataFrame(index=df.index)
    for col in df.columns:
        enum = enumerations.get(col)
        if not enum:
            df2[col] = df[col]
            continue
        convert = {v: i for i, v in enumerate(enum)}
        v_nan = convert.get('nan')
        vals = list(df[col])
        print('Convert %r : %d' % (col, len(convert)))
        if v_nan is None:
            for v in vals:
                assert v in convert, (v, enum[:10])
        categorical = [convert.get(v, v_nan) for v in vals]
        print(len(categorical), sorted(set(categorical))[:20])
        # df2[col] = pd.Series(categorical, dtype=np.int32).astype(np.int32)
        df2[col] = pd.Series(categorical, index=df.index)
        print(len(df2[col]), sorted(set(df2[col].values))[:20])
        df2[col] = df2[col].astype(np.int32)
        print('%20s : %20s -> %20s' % (col, df[col].dtype, df2[col].dtype))
    return df2


def compute_categoricals(name, df, threshold=20):
    print('=' * 80)
    print('categorical: %s' % name)
    print(list(df.columns))
    col_level = {}
    categoricals = []
    single_val_cols = []
    enumerations = {}

    for i, col in enumerate(df.columns):
        scores = df[col]
        print('%4d: %-20s %-10s - ' % (i, col, scores.dtype), end='', flush=True)
        try:
            levels = scores.unique()
            print('. ', end='', flush=True)
        except Exception as e:
             print(e)
             raise
        print('%7d levels %7d total %4.1f%%' % (len(levels), len(scores),
            100.0 * len(levels) / len(scores)), end='', flush=True)

        col_level[col] = len(levels)

        if len(levels) == 1:
            single_val_cols.append(col)

        if len(levels) <= threshold:
            level_counts = scores.value_counts()
            print(' ;', end='', flush=True)

            # levels.sort(key=lambda l: (-level_counts[l], l))
            enumerations[col] = level_counts.values

        print(' ***', flush=True)

        if len(levels) <= threshold:
            categoricals.append(col)
            for l, m in level_counts.iteritems():
                print('%20s: %7d %6.1f%%' % (l, m, 100.0 * m / len(scores)), flush=True)

    print('-' * 80)
    print('categoricals: %d %s' % (len(categoricals), categoricals))
    return col_level, categoricals, single_val_cols, enumerations
