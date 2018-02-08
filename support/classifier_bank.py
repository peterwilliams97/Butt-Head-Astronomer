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
import numpy as np
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


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

metrics = ['accuracy', 'f1', 'roc_auc']
metric_funcs = [accuracy_score, f1_score, roc_auc_score]


def compute_score_classifier(clf, X_train, y_train, X_test, y_test):

    t0 = time.clock()
    clf.fit(X_train, np.ravel(y_train))
    dt_fit = time.clock() - t0
    print('(%.1f sec - ' % dt_fit, end='', flush=True)

    t0 = time.clock()
    y_pred = clf.predict(X_test)
    dt_pred = time.clock() - t0
    print('%.1f sec) ' % dt_pred, flush=True)

    scores = []
    for met, score_func in zip(metrics, metric_funcs):
        score = score_func(y_test, y_pred)
        print('%20s %.4f' % (met, score), flush=True)
        scores.append(score)
    print('** %s' % scores, flush=True)
    return scores


def compute_score(X_train, y_train, X_test, y_test, n_neighbors):
    # clf = KNeighborsClassifier(n_neighbors)
    # clf = LogisticRegression(C=.1, dual=True)
    clf = XGBClassifier()
    scores = compute_score_classifier(clf, X_train, y_train, X_test, y_test)
    return scores


def compute_all_scores(X_train, y_train, X_test, y_test):

    classifier_score = {}
    for name, clf in zip(names, classifiers):
        print('** %20s: ' % name, end='', flush=True)
        scores = compute_score_classifier(clf, X_train, y_train, X_test, y_test)
        classifier_score[name] = scores

    return classifier_score
