# -*- coding: utf-8 -*-

from proc_mail import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
from sklearn.grid_search import GridSearchCV
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import os
import numpy as np

import json

load_if_possible = True


def save_source(x_train, y_train, vocabulary_train, x_test, y_test, vocabulary_test):
    with open("source_data.json", mode="w", encoding="utf-8") as f_out:
        json.dump({"x_train": x_train.tolist(), "vocabulary_train": vocabulary_train, "y_train": y_train.tolist(),
                   "x_test": x_test.tolist(), "vocabulary_test": vocabulary_test, "y_test": y_test.tolist()}, f_out)


def load_source():
    with open("source_data.json", encoding="utf-8") as f_in:
        source = json.load(f_in)
        return np.array(source["x_train"]), np.array(source["y_train"]), np.array(source["vocabulary_train"]), \
               np.array(source["x_test"]), np.array(source["y_test"]), np.array(source["vocabulary_test"])


def randomize_split(x, y):
    rv_xs = []
    rv_ys = []
    spliter = StratifiedShuffleSplit(y, 10, 0.1, random_state=0)
    for train_index, test_index in spliter:
        rv_xs.append({"train": x[train_index], "test": x[test_index]})
        rv_ys.append({"train": y[train_index], "test": y[test_index]})

    return rv_xs, rv_ys


def cal_score(model, xs, ys):
    score = []
    for i in range(0, len(xs)):
        score.append(model.score(xs[i]["test"], ys[i]["test"]))
    return score


def get_names(_dir):
    """
    Get the full name (with relative path) of the mail files.
    """
    names = []
    for root_dir, _, file_names in os.walk(_dir):
        for f in file_names:
            names.append(os.path.join(root_dir, f))

    return names


def get_mail_list(_filenames, _mail_list):
    """
    Add the files to the mail list.
    """
    for f in _filenames:
        _mail_list.append(process_mail(f))

    return _mail_list


def get_partial_vocabulary(_voca_file):
    with open(_voca_file) as f:
        partial_voca = f.read().split()
    return partial_voca


def feature_extraction_with_vocabulary(_mail_list, _voca_file):
    """
    stop-word filtering, and stemming using vectorizer in scikit learn.
    No need to lower case the mails as vectorizer will automatically do that.

    Note that we use the *specified partial* vocabulary _voca_file in this function
    """
    ### Please implement this function
    stemmer = PorterStemmer()
    stemmed_all = []
    for i in range(0, len(_mail_list)):
        print(i)
        mail_tokenized = word_tokenize(_mail_list[i])
        stemmed = []
        for token in mail_tokenized:
            stemmed.append(stemmer.stem(token))
        stemmed_all.append(' '.join(stemmed))
    partial_voca = get_partial_vocabulary(_voca_file)

    vectorizer = TfidfVectorizer(stop_words='english', vocabulary=partial_voca)

    feature_vectors = vectorizer.fit_transform(stemmed_all).todense()
    vocabulary = vectorizer.vocabulary_

    return feature_vectors, vocabulary


def main():
    train_dir = 'training'
    good_dir = os.path.join(train_dir, 'good')
    spam_dir = os.path.join(train_dir, 'spam')
    vocabulary_file = 'partial_vocabulary.txt'

    good_mail_names = get_names(good_dir)
    spam_mail_names = get_names(spam_dir)

    good_len = len(good_mail_names)
    spam_len = len(spam_mail_names)

    training_mails = []
    # process the good mails
    training_mails = get_mail_list(good_mail_names, training_mails)
    # process the spams
    training_mails = get_mail_list(spam_mail_names, training_mails)

    ### Obtain label vector y
    y = np.concatenate([np.ones(len(good_mail_names), dtype=np.int), np.zeros(len(spam_mail_names), dtype=np.int)])

    ### Get feature matrix Xtrain from the mails
    try:
        x_train_transform, y_train, vocabulary_train, x_test_transform, y_test, vocabulary_test = load_source()
    except (ValueError, FileNotFoundError):
        x_train, x_test, y_train, y_test = train_test_split(training_mails, y, test_size=0.1, random_state=0,
                                                            stratify=y)
        x_train_transform, vocabulary_train = feature_extraction_with_vocabulary(x_train, vocabulary_file)
        x_test_transform, vocabulary_test = feature_extraction_with_vocabulary(x_test, vocabulary_file)
        try:
            save_source(x_train_transform, y_train, vocabulary_train, x_test_transform, y_test, vocabulary_test)
        except Exception:
            pass

    ### Split data set
    cv = StratifiedShuffleSplit(y_train, 10, test_size=0.1, random_state=1)

    ### train your classifiers

    # Given the training data [X,y], you are required to build four classifiers using scikit-learn.
    # naive bayes
    # logistic regression
    """
    lr_model = LogisticRegression()
    lr_model.fit_transform(x_train, y_train)
    print("Logistic Regression: " + str(lr_model.score(x_validate, y_validate)))
    nb_model = GaussianNB()
    nb_model.fit(x_train, y_train)
    print("Naive Bayesian: " + str(nb_model.score(x_validate, y_validate)))
    # SVM
    svm_model = SVC()
    svm_model.fit(x_train, y_train)
    print("SVM: " + str(svm_model.score(x_validate, y_validate)))
    # random forest
    rf_model = RandomForestClassifier()
    rf_model.fit_transform(x_train, y_train)
    print("Random Forest: " + str(rf_model.score(x_validate, y_validate)))
    """
    """
    svm_model = SVC(kernel="linear")
    Cs = np.logspace(-4, 2, 3000)
    clf = GridSearchCV(svm_model, {"C": Cs}, n_jobs=-1, cv=cv)
    clf.fit(x_train_transform, y_train)
    print(clf.best_score_)
    print(clf.best_params_)
    print(clf.best_estimator_)
    """
    """
    lr_model = LogisticRegression(random_state=3)
    penalty = ['l1', 'l2']
    Cs = np.logspace(-4, 2, 200)
    solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag']
    clf = GridSearchCV(lr_model, {'C': Cs, "solver": solver, "penalty": penalty}, n_jobs=-1, cv=cv, error_score=0)
    clf.fit(x_train_transform, y_train)
    print(clf.best_score_)
    print(clf.best_params_)
    print(clf.best_estimator_)
    """
    rf_model = RandomForestClassifier(random_state=10)
    n_estimators = list(range(1, 20))
    criterion = ['gini', 'entropy']
    max_features = np.linspace(0.5, 1, 30)
    min_weight_fraction_leaf = np.linspace(0, 0.05, 10)
    clf = GridSearchCV(rf_model, {"n_estimators": n_estimators, "criterion": criterion, "max_features": max_features, "min_weight_fraction_leaf": min_weight_fraction_leaf}, n_jobs=-1, cv=cv, error_score=0)
    clf.fit(x_train_transform, y_train)
    print(clf.best_score_)
    print(clf.best_params_)
    print(clf.best_estimator_)
    print(clf.score(x_test_transform, y_test))


if __name__ == '__main__':
    main()
