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


def dim2(x):
    return '%s %g - %g - %g' % (dim(x), x.min(), x.mean(), x.max())


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
    xprint('save_json: path=%s' % path)


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
    print('save_pickle: path=%r obj=%s' % (path, type(obj)), end='')
    with open(temp_pickle, 'wb') as f:
        pickle.dump(obj, f)
    rename(temp_pickle, path)
    print(' - saved')


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
    print('save_pickle_gzip: path=%r obj=%s' % (path, type(obj)), end='')
    with gzip.open(temp_pickle, 'wb') as f:
        pickle.dump(obj, f)
    rename(temp_pickle, path)
    print(' - saved')


def load_model(model_path, config_path):

    with open(config_path, 'rt') as f:
        model = model_from_json(f.read())
    weights = load_pickle(model_path)
    model.set_weights(weights)

    xprint('load_model: model_path=%r config_path=%r weights=%s' % (model_path, config_path,
        dim(weights)))
    return model


def save_model(model, model_path, config_path):
    xprint('save_model: model_path=%r config_path=%r ' % (model_path, config_path), end='')
    assert isinstance(model_path, str) and isinstance(config_path, str)
    assert config_path.endswith('.json'), config_path

    weights = model.get_weights()
    xprint('weights=%s' % dim(weights))

    save_pickle(model_path, weights)
    with open(config_path, 'wt') as f:
        f.write(model.to_json())


AUC_DELTA = 0.001
CHECK_MODE_IO = True


class RocAucEvaluation(Callback):
    """ROC AUC for CV in Keras see for details: https://gist.github.com/smly/d29d079100f8d81b905e
    """

    def __init__(self, validation_data=(), w_val=None, interval=1, model_path=None, config_path=None,
        epoch_path=None, epoch_key=None, do_prime=False):
        super(Callback, self).__init__()

        xprint('RocAucEvaluation: validation_data=%s interval=%s, model_path=%s, config_path=%s '
               'epoch_path=%s do_prime=%s' % (len(validation_data), interval,
                model_path, config_path, epoch_path, do_prime))

        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.w_val = w_val
        self.model_path = model_path
        self.config_path = config_path
        self.epoch_path = epoch_path
        self.epoch_key = epoch_key
        self.epochs = load_json(epoch_path, {})
        self.epoch0 = self.epochs.get(epoch_key, 0)
        self.best_auc = 0.0
        self.best_epoch = -1
        self.t0 = time.perf_counter()
        if do_prime:
            model = load_model(model_path, config_path)
            y_pred = model.predict(self.X_val, verbose=0)
            auc = roc_auc_score(self.y_val, y_pred)
            xprint('\nROC-AUC - epoch: {:d} - score: {:.6f}'.format(0, auc))
            self.best_auc = auc
            del model

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)

            # print('\non_epoch_end: X_val=%s' % dim2(self.X_val))
            # print('on_epoch_end: y_val=%s' % dim2(self.y_val))
            # print('on_epoch_end: y_pred=%s' % dim2(y_pred))

            auc = roc_auc_score(self.y_val, y_pred, sample_weight=self.w_val)
            xprint('\nROC-AUC - epoch: {:d} - score: {:.6f} {}'.format(epoch + 1, auc, self.epoch_key))
            logs['val_auc'] = auc
            dt = time.perf_counter() - self.t0
            self.t0 = time.perf_counter()

            if auc >= self.best_auc + AUC_DELTA:
                xprint('RocAucEvaluation.fit: auc=%.3f > best_auc=%.3f dt=%.1f sec' % (auc,
                    self.best_auc, dt))
                self.best_auc = auc
                self.best_epoch = epoch
                self.epochs[self.epoch_key] = self.epoch0 + epoch
                save_json(self.epoch_path, self.epochs)

                save_model(self.model, self.model_path, self.config_path)

                if CHECK_MODE_IO:
                    model = load_model(self.model_path, self.config_path)
                    y_pred = model.predict(self.X_val, verbose=0)
                    auc = roc_auc_score(self.y_val, y_pred)
                    xprint('\n****ROC-AUC - epoch: {:d} - score: {:.6f}'.format(0, auc))
                    self.best_auc = auc
                    del model
            else:
                 xprint('RocAucEvaluation.fit: No improvement best_epoch=%d best_auc=%.3f dt=%.1f sec' %
                    (self.best_epoch + 1, self.best_auc, dt))


class SaveAllEpochs(Callback):
    """Save weights at the end of every epoch
    """

    # @staticmethod
    # def epoch_dict(epoch_path, default={'epoch1': 0, 'epoch2': 0}):
    #     return load_json(epoch_path, default)

    def __init__(self, model_path, config_path, epoch_path):

        super(Callback, self).__init__()

        xprint('SaveAllEpochs: model_path=%s, config_path=%s epoch_path=%s' % (
            model_path, config_path, epoch_path))

        self.model_path = model_path
        self.config_path = config_path
        self.epoch_path = epoch_path
        self.epochs = load_json(self.epoch_path, {})
        restarting = self.epochs.get('epoch1', 0) > 0 or self.epochs.get('epoch2', 0) > 0
        marker = ' *** restarting' if restarting else ''
        xprint('SaveAllEpochs: starting epochs=%s%s' % (self.epochs, marker))

    def on_epoch_end(self, epoch, logs={}):
        save_model(self.model, self.model_path, self.config_path)
        self.epochs['epoch1'] += 1
        save_json(self.epoch_path, self.epochs)
        xprint('SaveAllEpochs: epochs=%s' % self.epochs)

    def last_epoch(self):
        return SaveAllEpochs.epoch_dict(self.epoch_path, self.epochs)['epoch1']


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
