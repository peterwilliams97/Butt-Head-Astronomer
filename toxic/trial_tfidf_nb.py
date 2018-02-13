# coding: utf-8
"""
    Framework for evaluating and submitting models to Kaggle Toxic Comment challenge
"""
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from framework import evaluate, make_submission
from clf_tfidf_nb import ClfTfidfNB


def get_clf():
    def get_est():
        # return XGBClassifier()
        return LogisticRegression(C=8, dual=True)

    return ClfTfidfNB(get_est, do_spacy=False)


if False:
    make_submission(get_clf, 'submissionxxx.csv')

if True:
    evaluate(get_clf)

"""
    LogisticRegression regex tokenization
    --------------------------------------------------------------------------------------------------------------
        0: auc=0.979 (toxic:0.978, severe_toxic:0.982, obscene:0.988, threat:0.972, insult:0.980, identity_hate:0.974)
        1: auc=0.982 (toxic:0.979, severe_toxic:0.983, obscene:0.989, threat:0.990, insult:0.982, identity_hate:0.971)
        2: auc=0.982 (toxic:0.979, severe_toxic:0.984, obscene:0.988, threat:0.980, insult:0.981, identity_hate:0.977)
        3: auc=0.980 (toxic:0.978, severe_toxic:0.984, obscene:0.988, threat:0.976, insult:0.980, identity_hate:0.972)
        4: auc=0.982 (toxic:0.979, severe_toxic:0.982, obscene:0.987, threat:0.989, insult:0.980, identity_hate:0.972)
     Mean: auc=0.981 (toxic:0.979, severe_toxic:0.983, obscene:0.988, threat:0.981, insult:0.980, identity_hate:0.973)
    --------------------------------------------------------------------------------------------------------------
    auc=0.981 +- 0.004 (0%) 0.015 (1%)

    LogisticRegression spacy tokenization
    --------------------------------------------------------------------------------------------------------------
        0: auc=0.978 (toxic:0.977, severe_toxic:0.980, obscene:0.988, threat:0.972, insult:0.979, identity_hate:0.974)
        1: auc=0.981 (toxic:0.979, severe_toxic:0.978, obscene:0.988, threat:0.990, insult:0.981, identity_hate:0.971)
        2: auc=0.980 (toxic:0.977, severe_toxic:0.981, obscene:0.987, threat:0.980, insult:0.981, identity_hate:0.973)
        3: auc=0.978 (toxic:0.977, severe_toxic:0.982, obscene:0.987, threat:0.972, insult:0.979, identity_hate:0.972)
        4: auc=0.981 (toxic:0.978, severe_toxic:0.981, obscene:0.987, threat:0.990, insult:0.980, identity_hate:0.970)
     Mean: auc=0.980 (toxic:0.978, severe_toxic:0.981, obscene:0.987, threat:0.981, insult:0.980, identity_hate:0.972)
    --------------------------------------------------------------------------------------------------------------
    auc=0.980 +- 0.005 (0%) 0.015 (2%)

    xgboost regex
    --------------------------------------------------------------------------------------------------------------
        0: auc=0.959 (toxic:0.934, severe_toxic:0.978, obscene:0.972, threat:0.968, insult:0.954, identity_hate:0.949)
        1: auc=0.958 (toxic:0.933, severe_toxic:0.971, obscene:0.970, threat:0.970, insult:0.956, identity_hate:0.949)
        2: auc=0.962 (toxic:0.931, severe_toxic:0.981, obscene:0.971, threat:0.976, insult:0.956, identity_hate:0.959)
        3: auc=0.958 (toxic:0.932, severe_toxic:0.973, obscene:0.970, threat:0.966, insult:0.955, identity_hate:0.952)
        4: auc=0.959 (toxic:0.935, severe_toxic:0.974, obscene:0.969, threat:0.972, insult:0.955, identity_hate:0.952)
     Mean: auc=0.959 (toxic:0.933, severe_toxic:0.975, obscene:0.971, threat:0.970, insult:0.955, identity_hate:0.952)
    --------------------------------------------------------------------------------------------------------------
    auc=0.959 +- 0.015 (2%) range=0.043 (4%)

    xgboost spacy
    --------------------------------------------------------------------------------------------------------------
        0: auc=0.957 (toxic:0.931, severe_toxic:0.976, obscene:0.970, threat:0.963, insult:0.952, identity_hate:0.948)
        1: auc=0.957 (toxic:0.931, severe_toxic:0.968, obscene:0.966, threat:0.974, insult:0.956, identity_hate:0.948)
        2: auc=0.960 (toxic:0.929, severe_toxic:0.979, obscene:0.969, threat:0.971, insult:0.954, identity_hate:0.958)
        3: auc=0.956 (toxic:0.928, severe_toxic:0.972, obscene:0.967, threat:0.965, insult:0.954, identity_hate:0.949)
        4: auc=0.959 (toxic:0.933, severe_toxic:0.974, obscene:0.969, threat:0.976, insult:0.955, identity_hate:0.948)
     Mean: auc=0.958 (toxic:0.930, severe_toxic:0.974, obscene:0.968, threat:0.970, insult:0.954, identity_hate:0.950)
    --------------------------------------------------------------------------------------------------------------
    auc=0.958 +- 0.015 (2%) range=0.043 (5%)
"""
