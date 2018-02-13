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


"""
    --------------------------------------------------------------------------------------------------------------
        0: auc=0.975 (toxic:0.973, severe_toxic:0.988, obscene:0.987, threat:0.955, insult:0.981, identity_hate:0.966)
        1: auc=0.974 (toxic:0.972, severe_toxic:0.987, obscene:0.987, threat:0.956, insult:0.981, identity_hate:0.959)
        2: auc=0.975 (toxic:0.972, severe_toxic:0.989, obscene:0.988, threat:0.959, insult:0.980, identity_hate:0.964)
        3: auc=0.974 (toxic:0.972, severe_toxic:0.988, obscene:0.988, threat:0.952, insult:0.981, identity_hate:0.963)
        4: auc=0.978 (toxic:0.973, severe_toxic:0.988, obscene:0.986, threat:0.975, insult:0.980, identity_hate:0.965)
     Mean: auc=0.975 (toxic:0.972, severe_toxic:0.988, obscene:0.987, threat:0.960, insult:0.981, identity_hate:0.963)
    --------------------------------------------------------------------------------------------------------------
    auc=0.975 +- 0.011 (1%) range=0.029 (3%)

"""
