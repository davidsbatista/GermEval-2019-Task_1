#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import sys
import random

# import spacy
import time

from bs4 import BeautifulSoup, Tag

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline


def load_data():
    print("Pre-processing data")
    soup = BeautifulSoup(open(sys.argv[1]), "html.parser")
    data_x = []
    data_y = []
    for book in soup.findAll("book"):
        x = {'title': book.title.text,
             'body': book.body.text,
             'authors': book.authors.text,
             'published': book.published.text,
             'isbn': book.isbn.text}

        y = [t.text for categ in book.categories for t in categ if isinstance(t, Tag)]
        data_x.append(x)
        data_y.append(y)

    print(f'Processed {len(data_x)} documents')

    return data_x, data_y


def split_data(data_x, data_y):
    # split data, have a hold-out set of 20%
    n_samples = int(len(data_x) * 0.2)
    hold_out_idx = random.sample(range(0, len(data_x)), n_samples)
    hold_out_x = []
    hold_out_y = []
    train_x = []
    train_y = []
    for idx, (x, y) in enumerate(zip(data_x, data_y)):
        if idx in hold_out_idx:
            hold_out_x.append(x)
            hold_out_y.append(y)
        else:
            train_x.append(x)
            train_y.append(y)
    print("all data: ", len(data_x))
    print("train   : ", len(train_x))
    print("holdout : ", len(hold_out_x))

    return train_x, train_y, hold_out_x, hold_out_y


def tokenize(train_x):
    nlp = spacy.load('de_core_news_sm')
    all_docs = []
    c = 0
    print("\nBuild docs. for TF-IDF weighted vectors")
    for x in train_x:
        tokens = []
        doc = nlp(x['title'])
        for token in doc:
            tokens.append(token)
        doc = nlp(x['body'])
        for token in doc:
            tokens.append(token)
        all_docs.append(tokens)
        c += 1
        if c % 1000 == 0:
            print(c)

    print(len(all_docs))

    return all_docs


def main():

    # load a pre-process
    data_x, data_y = load_data()

    from sklearn.preprocessing import MultiLabelBinarizer
    ml_binarizer = MultiLabelBinarizer()
    y_labels = ml_binarizer.fit_transform(data_y)

    print('Total of {len(ml_binarizer.classes_} classes')

    data_x = data_x[:100]
    data_y = y_labels[:100]
    new_data_x = [x['title'] + "SEP" + x['body'] for x in data_x]

    # split into train and hold out set
    # train_x, train_y, hold_out_x, hold_out_y = split_data(new_data_x, data_y)
    train_x, test_x, train_y, test_y = train_test_split(new_data_x, data_y, test_size=0.20)

    # ToDo: set a baseline
    # build TF-IDF weighted vectors
    # all_docs = tokenize(train_x)

    # apply logistic regression with multi-label

    # TfidfVectorizer(stop_words=stop_words))
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1))
    ])
    parameters = {
        'tfidf__max_df': (0.25, 0.5, 0.75),
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        "clf__estimator__C": [0.01, 0.1, 1],
        "clf__estimator__class_weight": ['balanced', None],
    }

    grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=2, verbose=10)
    grid_search_tune.fit(train_x, train_y)

    print("Best parameters set:")
    print(grid_search_tune.best_estimator_.steps)

    # measuring performance on test set
    print("Applying best classifier on test data:")
    best_clf = grid_search_tune.best_estimator_
    predictions = best_clf.predict(test_x)
    print(classification_report(test_y, predictions, target_names=ml_binarizer.classes_))

if __name__ == '__main__':
    main()
