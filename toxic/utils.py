# coding: utf-8
"""
"""
import json
import pickle
import os


def load_json(path):
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


def load_pickle(path):
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
