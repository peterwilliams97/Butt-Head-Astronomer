# coding: utf-8
"""
    Another Keras solution to Kaggle Toxic Comment challenge
"""
from framework import evaluate, make_submission
from clf_lstm_glove import ClfLstmGlove


def get_clf():
    return ClfLstmGlove()


if False:
    make_submission(get_clf, 'lstm_glove.csv', remove_unknowns=False)

if True:
    evaluate(get_clf)


"""
    --------------------------------------------------------------------------------------------------------------
        0: auc=0.979 (toxic:0.974, severe_toxic:0.987, obscene:0.986, threat:0.969, insult:0.982, identity_hate:0.977)
        1: auc=0.982 (toxic:0.973, severe_toxic:0.985, obscene:0.987, threat:0.984, insult:0.982, identity_hate:0.978)
        2: auc=0.982 (toxic:0.976, severe_toxic:0.989, obscene:0.988, threat:0.982, insult:0.982, identity_hate:0.977)
        3: auc=0.981 (toxic:0.974, severe_toxic:0.987, obscene:0.987, threat:0.979, insult:0.982, identity_hate:0.979)
        4: auc=0.982 (toxic:0.975, severe_toxic:0.987, obscene:0.985, threat:0.983, insult:0.982, identity_hate:0.982)
     Mean: auc=0.981 (toxic:0.974, severe_toxic:0.987, obscene:0.987, threat:0.979, insult:0.982, identity_hate:0.979)
    --------------------------------------------------------------------------------------------------------------
    auc=0.981 +- 0.004 (0%) range=0.013 (1%)
"""
