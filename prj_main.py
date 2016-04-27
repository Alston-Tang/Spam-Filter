# -*- coding: utf-8 -*-

from proc_mail import *
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import os


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

    return feature_vectors, vocabulary

def main():
    train_dir = 'training'
    good_dir = os.path.join(train_dir, 'good')
    spam_dir = os.path.join(train_dir, 'spam')
    vocabulary_file = 'partial_vocabulary.txt'

    good_mail_names = get_names(good_dir)
    spam_mail_names = get_names(spam_dir)

    training_mails = []
    # process the good mails
    training_mails = get_mail_list(good_mail_names, training_mails)
    # process the spams
    training_mails = get_mail_list(spam_mail_names, training_mails)

    ### Get feature matrix Xtrain from the mails

    ### Obtain label vector y

    ### train your classifiers

    # Given the training data [X,y], you are required to build four classifiers using scikit-learn.
    # naive bayes
    # logistic regression
    # SVM
    # random forest

if __name__ == '__main__':
    main()
