# coding: utf-8
"""
"""
import json
import pickle
import gzip
import os
import datetime
import sys
import time
import numpy as np
import keras.backend as K
from keras.callbacks import Callback
from keras.models import model_from_json
from sklearn.metrics import roc_auc_score


def is_windows():
    return os.name == 'nt'


COMMENT = 'comment_text'
DATA_ROOT = 'd:\\data' if is_windows() else '~/data'
DATA_ROOT = os.path.expanduser(DATA_ROOT)
LOG_DIR = 'logs'

assert os.path.exists(DATA_ROOT), DATA_ROOT


xprint_f = None
xprint_path = None


def xprint_init(name, do_submisision):
    global xprint_f, xprint_path

    if do_submisision:
        name = '%s.submission' % name

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
    xprint('Starting log: %s name=%r path=%r prog=%r' % (now, name, xprint_path, sys.argv[0]))


def xprint(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except Exception as e:
        print('!!xprint failed. e=%s' % e)
        for a in args:
            print('!!', a)
        for k in sorted(kwargs):
            print('!! %s: %s' % (k, kwargs[k]))
        raise
    if xprint_f is None:
        return
    kwargs['file'] = xprint_f
    try:
        print(*args, **kwargs)
    except Exception as e:
        print('!!xprint failed. e=%s' % e)
        for a in args:
            print('!!', a)
        for k in sorted(kwargs):
            print('!! %s: %s' % (k, kwargs[k]))
        raise

    xprint_f.flush()


if False:
    xprint_init('blah', False)
    xprint('What do you think of htis')
    xprint('het', end=' ')
    xprint('teh')
    assert False


def _dim(x):
    try:
        return '%s:%s' % (list(x.shape), x.dtype)
    except:
        pass
    try:
        return list(x.shape)
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


def rename(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    os.renames(src, dst)


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
    rename(temp_json, path)


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


def touch(path):
    save_json(path, {'path': path})


temp_pickle = 'temp.pkl'


def save_pickle(path, obj):
    with open(temp_pickle, 'wb') as f:
        pickle.dump(obj, f)
    rename(temp_pickle, path)


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
    rename(temp_pickle, path)


def load_model(model_path, config_path, frozen, get_embeddings):
    assert frozen == (get_embeddings is not None), (model_path, config_path, frozen, get_embeddings)

    with open(config_path, 'rt') as f:
        model = model_from_json(f.read())
    weights = load_pickle(model_path)

    if frozen:
        embeddings = get_embeddings()
        weights = [embeddings] + weights
    model.set_weights(weights)

    xprint('load_model: model_path=%s frozen=%s weights=%s' % (model_path, frozen, dim(weights)))
    return model


def save_model(model, model_path, config_path, frozen):
    print('save_model: model_path=%r config_path=%r frozen=%s ' % (model_path, config_path, frozen), end='')
    assert isinstance(model_path, str) and isinstance(config_path, str) and isinstance(frozen, bool)
    assert config_path.endswith('.json'), config_path

    weights = model.get_weights()
    print('weights=%s' % dim(weights))
    if frozen:
        weights = weights[1:]

    save_pickle(model_path, weights)
    with open(config_path, 'wt') as f:
        f.write(model.to_json())


AUC_DELTA = 0.001


class RocAucEvaluation(Callback):
    """ROC AUC for CV in Keras see for details: https://gist.github.com/smly/d29d079100f8d81b905e
    """

    def __init__(self, validation_data=(), interval=1, model_path=None, config_path=None,
        frozen=False, was_frozen=True, get_embeddings=None, do_prime=False):
        super(Callback, self).__init__()

        print('validation_data=%s interval=%s, model_path=%s, config_path=%s '
              'frozen=%s was_frozen=%s get_embeddings=%s, do_prime=%s' % (
            len(validation_data), interval, model_path, config_path,
            frozen, was_frozen, get_embeddings, do_prime))

        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.model_path = model_path
        self.config_path = config_path
        self.frozen = frozen
        self.best_auc = 0.0
        self.best_epoch = -1
        self.top_weights = None
        self.t0 = time.clock()
        if do_prime:
            model = load_model(model_path, config_path, was_frozen, get_embeddings)
            y_pred = model.predict(self.X_val, verbose=0)
            auc = roc_auc_score(self.y_val, y_pred)
            xprint('\nROC-AUC - epoch: {:d} - score: {:.6f}'.format(0, auc))
            self.best_auc = auc
            del model

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            auc = roc_auc_score(self.y_val, y_pred)
            xprint('\nROC-AUC - epoch: {:d} - score: {:.6f}'.format(epoch + 1, auc))
            logs['val_auc'] = auc
            dt = time.clock() - self.t0
            self.t0 = time.clock()

            if auc >= self.best_auc + AUC_DELTA:
                xprint('RocAucEvaluation.fit: auc=%.3f > best_auc=%.3f dt=%.1f sec' % (auc,
                    self.best_auc, dt))
                self.best_auc = auc
                self.best_epoch = epoch

                weights = self.model.get_weights()
                self.top_weights = weights[1:]
                save_model(self.model, self.model_path, self.config_path, self.frozen)
            else:
                 xprint('RocAucEvaluation.fit: No improvement best_epoch=%d best_auc=%.3f dt=%.1f sec' %
                    (self.best_epoch + 1, self.best_auc, dt))


class SaveAllEpochs(Callback):
    """Save weights at the end of every epoch
    """

    @staticmethod
    def epoch_dict(epoch_path, default={'epoch1': 0, 'epoch2': 0}):
        return load_json(epoch_path, default)

    def __init__(self, model_path, config_path, epoch_path, frozen):

        super(Callback, self).__init__()

        xprint('SaveAllEpochs: model_path=%s, config_path=%s epoch_path=%s frozen=%s' % (
            model_path, config_path, epoch_path, frozen))

        self.model_path = model_path
        self.config_path = config_path
        self.epoch_path = epoch_path
        self.frozen = frozen
        self.epochs = load_json(self.epoch_path, {'epoch1': 0, 'epoch2': 0})
        restarting = self.epochs['epoch1'] > 0 or self.epochs['epoch2'] > 0
        marker = ' *** restarting' if restarting else ''
        xprint('SaveAllEpochs: starting epochs=%s%s' % (self.epochs, marker))

    def on_epoch_end(self, epoch, logs={}):
        save_model(self.model, self.model_path, self.config_path, self.frozen)
        if self.frozen:
            self.epochs['epoch1'] += 1
        else:
            self.epochs['epoch2'] += 1
        save_json(self.epoch_path, self.epochs)
        xprint('SaveAllEpochs: epochs=%s' % self.epochs)

    def last_epoch1(self):
        return SaveAllEpochs.epoch_dict(self.epoch_path, self.epochs)['epoch1']

    def last_epoch2(self):
        return SaveAllEpochs.epoch_dict(self.epoch_path, self.epochs)['epoch2']


class Cycler:
    """A Cycler object cycles through `items` forever in sizes of `batch_size`
        e.g.
            c = Cycler([0, 1, 2], 2)
            c.batch() -> 0 1
            c.batch() -> 2 0
            c.batch() -> 1 2
            ...
    """

    def __init__(self, items, batch_size):
        self.items = items
        self.batch_size = batch_size
        self.i = 0

    def batch(self):
        i1 = (self.i + self.batch_size) % len(self.items)
        if self.i < i1:
            items = self.items[self.i:i1]
        else:
            items = np.concatenate((self.items[self.i:], self.items[:i1]))
        self.i = i1
        return items
