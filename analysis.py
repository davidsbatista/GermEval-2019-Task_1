#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import load_data


def top_words_per_class(train_data_x, train_data_y):

    # see: https://buhrmann.github.io/tfidf-analysis.html
    def display_scores(vectorizer, tfidf_result):
        """
        taken from http://stackoverflow.com/questions/16078015/
        """
        scores = zip(vectorizer.get_feature_names(),
                     np.asarray(tfidf_result.sum(axis=0)).ravel())
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        for item in sorted_scores[:25]:
            print("{0:50} Score: {1}".format(item[0], item[1]))

    # top-words per class
    top_words_label = defaultdict(list)
    for sample_x, sample_y in zip(train_data_x, train_data_y):
        new_data_x = sample_x['title'] + " SEP " + sample_x['body']
        top_words_label[sample_y[0][0]].append(new_data_x)

    # compute TF-IDF vectorizer for each label
    print("\nComputing TF-IDF vectors for each label")
    tf_idf_labels = defaultdict()
    for k, v in top_words_label.items():
        print(k)
        tfidf = TfidfVectorizer(stop_words=set(stopwords.words('german')),
                                ngram_range=(1, 2), max_df=0.75)
        result = tfidf.fit_transform(v)
        tf_idf_labels[k] = (tfidf, result)
    for k, v in tf_idf_labels.items():
        print(k)
        display_scores(v[0], v[1])
        print()


def extract_hierarchy():
    # extract hierarchical structure
    level_0 = ['Architektur & Garten',
               'Ganzheitliches Bewusstsein',
               'Glaube & Ethik',
               'Kinderbuch & Jugendbuch',
               'Künste',
               'Literatur & Unterhaltung',
               'Ratgeber',
               'Sachbuch']

    hierarchical_level_0 = defaultdict(list)
    hierarchical_level_1 = defaultdict(list)
    all_lines = []

    with open('blurbs_dev_participants/hierarchy.txt', 'rt') as f_in:
        for line in f_in:
            all_lines.append(line)
            parts = line.split('\t')
            if any(x == parts[0].strip() for x in level_0):
                hierarchical_level_1[parts[1].strip()] = []
                hierarchical_level_0[parts[0].strip()].append(parts[1].strip())
    level_1 = list(hierarchical_level_1.keys())

    for line in all_lines:
        parts = line.split('\t')
        if any(x == parts[0].strip() for x in level_1):
            hierarchical_level_1[parts[0].strip()].append(parts[1].strip())

    print("\nLevel 0")
    for k, v in hierarchical_level_0.items():
        print(k, '\t', len(v))
    print("\nLevel 1")

    for k, v in hierarchical_level_1.items():
        print(k, '\t', len(v))
    print()

    for k, v in hierarchical_level_0.items():
        print(k)
        print("-" * len(k))
        for label in v:
            print(label)
        print()
    print()

    for k, v in hierarchical_level_1.items():
        print(k)
        print("-" * len(k))
        for label in v:
            print(label)
        print()


def data_analysis(train_data_x, train_data_y, labels):

    top_words_per_class(train_data_x, train_data_y)

    # for sample_x, sample_y in zip(train_data_x, train_data_y):
    #     # new_data_x = [x['title'] + " SEP " + x['body'] for x in train_data_x]
    #     print(sample_x['authors'].split(","))
    #     print(sample_y)
    #     print()
    #
    #     for x in sample_x['authors'].split(","):
    #         author_topic[x.strip()] += 1
    #
    # for k, v in author_topic.items():
    #     if v > 1:
    #         print(k, v)

    # extract_hierarchy()

    author_topic = defaultdict(int)

    """
    from pandas import DataFrame
    df_stats_level_0 = DataFrame.from_dict(labels['0'], orient='index', columns=['counts'])

    print(df_stats_level_0)
    df_stats_level_0.plot(y='counts', kind='bar', legend=False, grid=True, figsize=(15, 8))
    print()

    df_stats_level_1 = DataFrame.from_dict(labels['1'], orient='index', columns=['counts'])
    print(df_stats_level_1)
    df_stats_level_1.plot(y='counts', kind='bar', legend=False, grid=True, figsize=(15, 8))
    print()

    df_stats_level_2 = DataFrame.from_dict(labels['2'], orient='index', columns=['counts'])
    print(df_stats_level_2)
    df_stats_level_2.plot(y='counts', kind='bar', legend=False, grid=True, figsize=(15, 8))
    """


def main():
    # load train data
    train_data_x, train_data_y, labels = load_data('blurbs_train.txt')

    # load dev data
    dev_data_x, _, _ = load_data('blurbs_dev_participants.txt')

    # do some data analysis
    data_analysis(train_data_x, train_data_y, labels)

if __name__ == '__main__':
    main()
