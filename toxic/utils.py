# coding: utf-8
"""
"""
import json
import pickle
import gzip
import os
import datetime
import keras.backend as K


def is_windows():
    return os.name == 'nt'


COMMENT = 'comment_text'
DATA_ROOT = 'd:\\data' if is_windows() else '~/data'
DATA_ROOT = os.path.expanduser(DATA_ROOT)
LOG_DIR = 'logs'

assert os.path.exists(DATA_ROOT), DATA_ROOT


xprint_f = None
xprint_path = None


def xprint_init(name):
    global xprint_f, xprint_path

    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, '%s.log' % name)
    # assert not os.path.exists(path), path
    xprint_path = path

    if xprint_f is not None:
        xprint_f.close()

    xprint_f = open(xprint_path, 'at')
    assert xprint_f, xprint_path

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    xprint('@' * 80)
    xprint('Starting log: %s %r %r' % (now, name, xprint_path))


def xprint(*args):
    print(*args)
    if xprint_f is None:
        return
    print(*args, file=xprint_f)
    xprint_f.flush()


if False:
    xprint_init('blah')
    xprint('What do you think of htis')


def _dim(x):
    try:
        return '%s:%s' % (list(x.shape), x.dtype)
    except:
        pass
    try:
        return K.int_shape(x)
    except:
        pass
    return type(x)


def dim(x):
    if isinstance(x, (list, tuple)):
        return '%s:%d %s' % (type(x), len(x), [_dim(z) for z in x[:10]])
    return _dim(x)


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


def load_pickle_gzip(path, default=None):
    if default is not None and not os.path.exists(path):
        return default
    try:
        with gzip.open(path, 'rb') as f:
            obj = pickle.load(f)
    except:
        print('load_pickle failed: path=%r' % path)
        raise
    return obj


temp_pickle_gzip = 'temp.pkl.gzip'


def save_pickle_gzip(path, obj):
    with gzip.open(temp_pickle, 'wb') as f:
        pickle.dump(obj, f)
    os.renames(temp_pickle, path)

