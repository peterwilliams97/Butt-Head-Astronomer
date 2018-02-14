# coding: utf-8
"""
    Another Keras solution to Kaggle Toxic Comment challenge
"""
from framework import evaluate, make_submission
from clf_lstm_glove import ClfLstmGlove


# Classifier parameters
embed_size = 50
maxlen = 100
max_features = 20000
epochs = 40
learning_rate = [0.007, 0.005, 0.002, 0.003, 0.000]
dropout = 0.1

submission_name = 'lstm_glove_%3d_%3d_%4d_%.3f.csv' % (embed_size, maxlen, max_features,
    dropout)


def get_clf():
    return ClfLstmGlove(embed_size=embed_size, maxlen=maxlen, max_features=max_features,
            dropout=dropout,
            epochs=epochs, learning_rate=learning_rate)


print(get_clf())
if False:
    make_submission(get_clf, submission_name)

if True:
    evaluate(get_clf, n=5)
print('embed_size, maxlen, max_features =', embed_size, maxlen, max_features)
print(get_clf())

"""
    embed_size = 50
    maxlen = 100
    max_features = 20000
    epochs = 40
    learning_rate = [0.007, 0.005, 0.002, 0.003, 0.000]
    dropout = 0.1
    lstm_glove_ 50_100_20000_0.100.csv ****

    embed_size = 50
    maxlen = 100
    max_features = 20000
    --------------------------------------------------------------------------------------------------------------
        0: auc=0.979 (toxic:0.974, severe_toxic:0.987, obscene:0.986, threat:0.969, insult:0.982, identity_hate:0.977)
        1: auc=0.982 (toxic:0.973, severe_toxic:0.985, obscene:0.987, threat:0.984, insult:0.982, identity_hate:0.978)
        2: auc=0.982 (toxic:0.976, severe_toxic:0.989, obscene:0.988, threat:0.982, insult:0.982, identity_hate:0.977)
        3: auc=0.981 (toxic:0.974, severe_toxic:0.987, obscene:0.987, threat:0.979, insult:0.982, identity_hate:0.979)
        4: auc=0.982 (toxic:0.975, severe_toxic:0.987, obscene:0.985, threat:0.983, insult:0.982, identity_hate:0.982)
     Mean: auc=0.981 (toxic:0.974, severe_toxic:0.987, obscene:0.987, threat:0.979, insult:0.982, identity_hate:0.979)
        0: auc=0.982 (toxic:0.976, severe_toxic:0.988, obscene:0.988, threat:0.981, insult:0.983, identity_hate:0.978)
        1: auc=0.984 (toxic:0.975, severe_toxic:0.987, obscene:0.988, threat:0.990, insult:0.983, identity_hate:0.979)
        0: auc=0.982 (toxic:0.976, severe_toxic:0.989, obscene:0.988, threat:0.975, insult:0.982, identity_hate:0.979)
        0: auc=0.981 (toxic:0.976, severe_toxic:0.988, obscene:0.988, threat:0.979, insult:0.982, identity_hate:0.976)
        0: auc=0.980 (toxic:0.976, severe_toxic:0.987, obscene:0.987, threat:0.970, insult:0.982, identity_hate:0.978)
    --------------------------------------------------------------------------------------------------------------
    auc=0.981 +- 0.004 (0%) range=0.013 (1%)

    embed_size = 300
    maxlen = 100
    max_features = 20000
       0: auc=0.980 (toxic:0.973, severe_toxic:0.987, obscene:0.986, threat:0.978, insult:0.980, identity_hate:0.978)
"""
