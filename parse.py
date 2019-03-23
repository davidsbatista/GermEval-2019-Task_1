#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import defaultdict
from os.path import join

from bs4 import BeautifulSoup, Tag

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from nltk.corpus import stopwords


def load_data(file):
    base_path = 'blurbs_dev_participants/'
    full_path = join(base_path, file)

    topics_distribution = defaultdict(int)

    with open(full_path, 'rt') as f_in:
        print("Loading {}".format(full_path))
        soup = BeautifulSoup(f_in, "html.parser")
        data_x = []
        data_y = []
        for book in soup.findAll("book"):
            x = {'title': book.title.text,
                 'body': book.body.text,
                 'authors': book.authors.text,
                 'published': book.published.text,
                 'isbn': book.isbn.text}

            if 'train' in full_path:
                # y = [t.text for categ in book.categories for t in categ if isinstance(t, Tag)]
                topics = set()
                for categ in book.categories:
                    for t in categ:
                        if isinstance(t, Tag):
                            if t['d'] == "0":
                                topics.add(t.text)
                                topics_distribution[t.text] += 1
                data_y.append(list(topics))

            data_x.append(x)

        print(f'Loaded {len(data_x)} documents')

    return data_x, data_y, topics_distribution


def main():

    """
    Subtasks
    This shared task consists of two subtask, described below. You can participate in one of
    them or both.

    Subtask A
    The task is to classify german books into one or multiple most general writing genres (d=0).
    Therfore, it can be considered a multi-label classification task. In total, there are 8 classes
    that can be assigned to a book:
    - Literatur & Unterhaltung,
    - Ratgeber,
    - Kinderbuch & Jugendbuch,
    - Sachbuch,
    - Ganzheitliches Bewusstsein,
    - Glaube & Ethik,
    - Künste,
    - Architektur & Garten.


    Subtask B
    The second task is a hierarchical multi-label classification into multiple writing genres.
    In addition to the very general writing genres additional genres of different specificity can
    be assigned to a book. In total, there are 343 different classes that are hierarchically
    structured.

    :return:
    """

    # load train data
    train_data_x, train_data_y, topics_distribution = load_data('blurbs_train.txt')

    for k, v in topics_distribution.items():
        print(k, v)

    # load dev data
    # dev_data_x, dev_data_y = load_data('blurbs_dev_participants.txt')

    # one-hot vectors on y labels
    ml_binarizer = MultiLabelBinarizer()
    y_labels = ml_binarizer.fit_transform(train_data_y)
    print('Total of {} classes'.format(len(ml_binarizer.classes_)))

    train_data_x = train_data_x[:1000]
    data_y = y_labels[:1000]

    new_data_x = [x['title'] + "SEP" + x['body'] for x in train_data_x]

    # split into train and hold out set
    train_x, test_x, train_y, test_y = train_test_split(new_data_x,
                                                        data_y,
                                                        random_state=42,
                                                        test_size=0.20)

    # set a baseline: build TF-IDF weighted vectors and apply
    # logistic regression with multi-label
    stop_words = set(stopwords.words('german'))
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stop_words)),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='sag', max_iter=1000), n_jobs=2))
    ])
    parameters = {
        'tfidf__max_df': (0.25, 0.5, 0.75),
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        "clf__estimator__C": [0.01, 0.1, 1],
        "clf__estimator__class_weight": ['balanced', None],
    }

    grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=2, verbose=2)

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
