# coding: utf-8
"""
    Framework for evaluating and submitting models to Kaggle Toxic Comment challenge
"""
from utils import label_cols
from framework import evaluate, make_submission
from clf_glove_nb_spacy import ClfGloveNBSpacy


clf = ClfGloveNBSpacy(label_cols, n_gram=2)

if False:
    make_submission(clf, 'submissionxxx.csv')

if True:
    evaluate(clf)


# make_submission(train, test)
evaluate(clf)
