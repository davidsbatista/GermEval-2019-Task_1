#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gzip
from collections import defaultdict
from os.path import join

from bs4 import BeautifulSoup, Tag, NavigableString

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from nltk.corpus import stopwords


def load_data(file, hierarchical=False):
    """
    Parses and loads the training/dev/test data into a list of dicts

    :param hierarchical:
    :param file:
    :return:
    """
    base_path = 'blurbs_dev_participants/'
    full_path = join(base_path, file)

    topics_distribution = defaultdict(int)  # level 0 - only

    labels_by_level = {'0': defaultdict(int),
                       '1': defaultdict(int),
                       '2': defaultdict(int)}

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
                if not hierarchical:
                    topics = set()
                    for categ in book.categories:
                        for t in categ:
                            if isinstance(t, Tag):
                                if t['d'] == "0":
                                    topics.add(t.text)
                                    topics_distribution[t.text] += 1
                    data_y.append(list(topics))

                elif hierarchical:
                    categories = []
                    for categ in book.categories:
                        if isinstance(categ, NavigableString):
                            continue
                        topics = [None] * 3
                        for t in categ:
                            if isinstance(t, Tag):
                                level = int(t['d'])
                                topics[level] = t.text
                                labels_by_level[str(level)][t.text] += 1
                        categories.append(topics)

                    data_y.append(categories)

            data_x.append(x)

        print(f'Loaded {len(data_x)} documents')

    if hierarchical:
        return data_x, data_y, labels_by_level
    else:
        return data_x, data_y, topics_distribution


def generate_submission_file(predictions, ml_binarizer, dev_data_x):
    """
    All submissions should be formatted as shown below (submissions to both tasks) and written
    into a file called answer.txt and uploaded as a zipped file:

    subtask_a
    ISBN<Tab>Label1<Tab>Label2<Tab>....<Tab>Label_n
    ISBN<Tab>Label1<Tab>Label2<Tab>....<Tab>Label_n...
    subtask_b
    ISBN<Tab>Label1<Tab>Label2<Tab>....<Tab>Label_n
    ISBN<Tab>Label1<Tab>Label2<Tab>....<Tab>Label_n...

    :return:
    """

    with gzip.open('answer.txt.zip', 'wt') as f_out:
        f_out.write(str('subtask_a\n'))
        for pred, data in zip(ml_binarizer.inverse_transform(predictions), dev_data_x):
            f_out.write(data['isbn']+'\t'+'\t'.join([p for p in pred])+'\n')


def train_model(train_data_x, train_data_y):
    """
    Set a simple baseline,

    - TF-IDF weighted vectors as data representation and apply logistic regression with multi-label

    :param train_data_x:
    :param train_data_y:
    :return: tuned classifier

    """
    # encode y labels into one-hot vectors
    ml_binarizer = MultiLabelBinarizer()
    y_labels = ml_binarizer.fit_transform(train_data_y)
    print('Total of {} classes'.format(len(ml_binarizer.classes_)))

    #train_data_x = train_data_x[:1000]
    #data_y = y_labels[:1000]
    data_y = y_labels

    new_data_x = [x['title'] + "SEP" + x['body'] for x in train_data_x]

    # split into train and hold out set
    train_x, test_x, train_y, test_y = train_test_split(new_data_x,
                                                        data_y,
                                                        random_state=42,
                                                        test_size=0.20)

    stop_words = set(stopwords.words('german'))
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stop_words)),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='sag', max_iter=3000), n_jobs=3))
    ])
    parameters = {
        'tfidf__max_df': (0.25, 0.5, 0.75),
        'tfidf__ngram_range': [(1, 1), (1,2), (1,3)],
        "clf__estimator__C": [0.01, 0.1, 1],
        "clf__estimator__class_weight": ['balanced', None],
    }
    grid_search_tune = GridSearchCV(pipeline, parameters, cv=3, n_jobs=3, verbose=2)
    grid_search_tune.fit(train_x, train_y)
    print("Best parameters set:")
    print(grid_search_tune.best_estimator_.steps)

    # measuring performance on test set
    print("Applying best classifier on test data:")
    best_clf = grid_search_tune.best_estimator_
    predictions = best_clf.predict(test_x)
    print(classification_report(test_y, predictions, target_names=ml_binarizer.classes_))

    return best_clf, ml_binarizer


def main():

    """
    Subtasks
    This shared task consists of two subtask, described below. You can participate in one of
    them or both.

    Subtask A
    =========
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
    =========
    The second task is a hierarchical multi-label classification into multiple writing genres.
    In addition to the very general writing genres additional genres of different specificity can
    be assigned to a book. In total, there are 343 different classes that are hierarchically
    structured.

    :return:
    """
    # load train data
    train_data_x, train_data_y, labels = load_data('blurbs_train.txt', hierarchical=True)

    # for k, v in topics_distribution.items():
    #     print(k, v)
    #
    # print()

    print(train_data_x[-1])
    print(train_data_y[-1])
    print()

    for x, y in zip(train_data_x, train_data_y):
        if x['isbn'] == '9783579087276':
            print(x)
            print(y)
            print()

    for k, v in labels.items():
        print(k)
        print(len(v))
        print(v)
        print()

    # best_clf, ml_binarizer = train_model(train_data_x, train_data_y)
    # print(best_clf)

    # apply on  dev data
    # dev_data_x, _, _ = load_data('blurbs_dev_participants.txt')
    # new_data_x = [x['title'] + "SEP" + x['body'] for x in dev_data_x]
    # predictions = best_clf.predict(new_data_x)
    # generate_submission_file(predictions, ml_binarizer, dev_data_x)


if __name__ == '__main__':
    main()
