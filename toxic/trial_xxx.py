# coding: utf-8
"""
    Keras solution to Kaggle Toxic Comment challenge
"""
from framework import evaluate, make_submission
from clf_keras import ClfKeras


def get_clf():
    return ClfKeras()


if False:
    make_submission(get_clf, 'keras_submission.csv', remove_unknowns=False)

if True:
    evaluate(get_clf, remove_unknowns=False)

