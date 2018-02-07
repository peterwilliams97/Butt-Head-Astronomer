#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=====================
Classifier comparison
=====================

A comparison of a several classifiers in scikit-learn on synthetic datasets.
The point of this example is to illustrate the nature of decision boundaries
of different classifiers.
This should be taken with a grain of salt, as the intuition conveyed by
these examples does not necessarily carry over to real datasets.

Particularly in high-dimensional spaces, data can more easily be separated
linearly and the simplicity of classifiers such as naive Bayes and linear SVMs
might lead to better generalization than is achieved by other classifiers.

The plots show training points in solid colors and testing points
semi-transparent. The lower right shows the classification accuracy on the test
set.
"""
import time
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier


names = [
    "Nearest Neighbors", "Logisitic",
    "Decision Tree", "Random Forest", "AdaBoost", "XGBoost",

    "Naive Bayes", "QDA"
    "RBF SVM", "Linear SVM",
    "Gaussian Process", "Neural Net",
    ]

classifiers = [
    KNeighborsClassifier(3),
    LogisticRegression(C=4, dual=True),

    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    XGBClassifier(),

    GaussianNB(),
    QuadraticDiscriminantAnalysis(),

    SVC(gamma=2, C=1),
    SVC(kernel="linear", C=0.025),

    GaussianProcessClassifier(1.0 * RBF(1.0)),
    MLPClassifier(alpha=1),
]


def compute_score(X, y, cv=2):

    metrics = ['accuracy', 'f1',
               #'neg_log_loss',
               'roc_auc']
    classifier_score = {}
    for name, clf in zip(names, classifiers):

        # clf.fit(X_train, y_train)
        scores = []
        for met in metrics:
            print('** %20s: %10s ' % (name, met), end='', flush=True)
            t0 = time.clock()
            score_cv = cross_val_score(clf, X, y, cv=cv, scoring=met)
            score = score_cv.mean()
            # score = clf.score(X_test, y_test, scoring=metric)
            dt = time.clock() - t0
            print('%20s %.4f (%.1f sec)' % (score_cv, score, dt), flush=True)
            scores.append(score)
        print('** %20s: %s' % (name, scores), flush=True)
        classifier_score[name] = scores

    return classifier_score, metrics
