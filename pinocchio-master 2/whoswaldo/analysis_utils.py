import csv

import graphviz
import matplotlib as mpl
from numpy.random import random_integers
import pandas
from pandas.plotting import scatter_matrix
from pandas import read_csv
from sklearn import model_selection
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from intake_utils import (classifyData, digestData, loadDataset)
from settings import ANALYSIS_FEATURES, ANALYSIS_NONSCORE_ROWS


def determinePairwiseCorr(filename):
    dataset = read_csv(filename, header=0, index_col=['rental-id'])
    print(dataset.corr())


def generateClassifierData(files, classifications, outputFile, header=False):
    if header:
        with open(outputFile, 'w+') as outfile:
            csvwriter = csv.writer(outfile)
            csvwriter.writerow(ANALYSIS_FEATURES)

    for i in range(len(files)):
        try:
            classifyData(files[i], outputFile, 'a+',
                         classification=classifications[i])
        except IndexError:
            print('Num files must equal num classifications!')
            return


def getCrossValidationScore(model, data, classifications, seed=None, scoring='accuracy'):
    if not seed:
        seed = random_integers(0, 100)

    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model(), data,
                                                 classifications,
                                                 cv=kfold, scoring=scoring)

    return cv_results


def evaluateModels(trainingData, trainingClassifications, output=True):
    """
    outputs results of testing various models for fitness
    of the training data
    """

    # random seed
    seed = random_integers(0, 100)
    scoring = 'accuracy'

    # load classifiers
    models = []
    models.append(('LR', LogisticRegression))
    models.append(('LDA', LinearDiscriminantAnalysis))
    models.append(('KNN', KNeighborsClassifier))
    models.append(('CART', DecisionTreeClassifier))
    models.append(('NB', GaussianNB))
    models.append(('SVM', SVC))

    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        cv_results = getCrossValidationScore(model, trainingData,
                                             trainingClassifications,
                                             seed=seed)

        results.append((cv_results, model))
        names.append(name)

        if output:
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)

    return results


def testClassifiers(infile, validation_size=0.20, output=True):
    """
    determines the best classifier for the training data
    """
    X, Y, _ = loadDataset(infile)
    classifiers_list = []

    for i in range(10):

        seed = random_integers(0, 100)
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
        results = evaluateModels(X_train, Y_train)

        # evaluate classifiers
        best_mean = 0
        best_std = 1
        best_classifier = None
        for res, model in results:
            if ((res.mean() > best_mean) or
                (res.mean() == best_mean and res.std() < best_std)):
                best_mean = res.mean()
                best_std = res.std()
                best_classifier = model

        classifiers_list.append(best_classifier)

        if output:
            print('\nBest Classifier: {}\nMean: {}'.format(best_classifier.__name__, best_mean))

            classifier = best_classifier()
            classifier.fit(X=X_train, y=Y_train)
            predictions = classifier.predict(X_validation)
            print('Accuracy Score: {}'.format(accuracy_score(Y_validation, predictions)))
            print('Confusion Matrix: \n{}'.format(confusion_matrix(Y_validation, predictions)))
            print('Classification Report: {}'.format(classification_report(Y_validation, predictions)))

    from collections import Counter
    results = dict(Counter(classifiers_list))
    import operator
    return max(results.items(), key=operator.itemgetter(1))[0]
