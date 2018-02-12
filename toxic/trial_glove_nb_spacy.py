# coding: utf-8
"""
    Framework for evaluating and submitting models to Kaggle Toxic Comment challenge
"""
from framework import label_cols, evaluate, make_submission
from clf_glove_nb_spacy import ClfGloveNBSpacy


def get_clf():
    return ClfGloveNBSpacy(n_gram=2)


if False:
    make_submission(clf, 'submissionxxx.csv')

if True:
    evaluate(get_clf, n=4)
    # evaluate(get_clf)
    # evaluate(get_clf)
    # evaluate(get_clf)
