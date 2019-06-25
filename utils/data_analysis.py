#!/usr/bin/env python
# -*- coding: utf-8 -*-

import statistics
from collections import defaultdict

import nltk
import numpy as np
from nltk import sent_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from pre_processing import load_data


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
    hierarchical_level_1 = defaultdict(list)
    hierarchical_level_2 = defaultdict(list)
    all_lines = []

    with open('data/blurbs_test_participants/hierarchy.txt', 'rt') as f_in:
        for line in f_in:
            all_lines.append(line)
            parts = line.split('\t')
            if any(x == parts[0].strip() for x in level_0):
                hierarchical_level_1[parts[0].strip()].append(parts[1].strip())
                hierarchical_level_2[parts[1].strip()] = []
    level_2 = list(hierarchical_level_2.keys())

    for line in all_lines:
        parts = line.split('\t')
        if any(x == parts[0].strip() for x in level_2):
            hierarchical_level_2[parts[0].strip()].append(parts[1].strip())

    return hierarchical_level_1, hierarchical_level_2


def stats_genres_per_blurb(train_data_y):
    genres_per_blurb = []
    genres_per_level_per_blurb = {0: [], 1: [], 2: []}
    samples_per_co_occurrence = defaultdict(int)
    nr_samples_leaf_at_level = {0: 0, 1: 0, 2: 0}
    for blurb in train_data_y:
        all_genres = set()
        genres_per_level = {0: set(), 1: set(), 2: set()}
        max_level = 0
        for labels in blurb:
            for k, v in labels.items():
                max_level = k if k > max_level else max_level
                all_genres.add(v)
                genres_per_level[k].add(v)
        nr_samples_leaf_at_level[max_level] += 1
        genres_per_blurb.append(len(all_genres))
        genres_per_level_per_blurb[0].append(len(genres_per_level[0]))
        genres_per_level_per_blurb[1].append(len(genres_per_level[1]))
        genres_per_level_per_blurb[2].append(len(genres_per_level[2]))
        samples_per_co_occurrence['_'.join(list(sorted(all_genres)))] += 1

    print()
    print("Avg. genres per label", statistics.mean(genres_per_blurb))
    print("Std. deviation : ", statistics.stdev(genres_per_blurb))
    print()
    print("Avg. genres per label per level")
    for x, v in genres_per_level_per_blurb.items():
        print("Level ", x)
        print(statistics.mean(v))
        print(statistics.stdev(v))
        print()

    print("Avg. blurb per co-occurrence")
    samples_co_occurrences = list(samples_per_co_occurrence.values())
    print(statistics.mean(samples_co_occurrences))
    print(statistics.stdev(samples_co_occurrences))

    print("\nLeaf nodes at each level (1;2;3)")
    print("1 :", nr_samples_leaf_at_level[0], nr_samples_leaf_at_level[0]/len(train_data_y))
    print("2 :", nr_samples_leaf_at_level[1], nr_samples_leaf_at_level[1]/len(train_data_y))
    print("3 :", nr_samples_leaf_at_level[2], nr_samples_leaf_at_level[2]/len(train_data_y))


def count_sentences_tokens(train_data_x):
    # the title of the book is considered a sentence
    new_data_x = [x['title'] + ". " + x['body'] for x in train_data_x]
    sentences_per_blurb = []
    tokens_per_blurb = []
    for x in new_data_x:
        doc_tokens = []
        sentences_per_blurb.append(len(sent_tokenize(x)))
        for s in sent_tokenize(x):
            tokens = wordpunct_tokenize(s)
            # consider only alphanumeric tokens
            words = [w.lower() for w in nltk.Text(tokens) if w.isalpha()]
            doc_tokens.extend(words)
        tokens_per_blurb.append(len(doc_tokens))
    return sentences_per_blurb, tokens_per_blurb


def data_analysis(train_data_x, train_data_y, test_data_x):

    # how many genres/classes per level
    hierarchical_level_1, hierarchical_level_2 = extract_hierarchy()
    print()
    print("top-level: ", len(hierarchical_level_1))
    level_1 = 0
    for k, v in hierarchical_level_1.items():
        level_1 += len(v)
    print("hierarchical_level_1:", level_1)
    level_2 = 0
    for k, v in hierarchical_level_2.items():
        level_2 += len(v)
    print("hierarchical_level_2:", level_2)
    print("\n")

    # compute quantitative statistics about the dataset
    sentences_per_blurb, tokens_per_blurb = count_sentences_tokens(train_data_x)
    print("Training data")
    print(len("Training data")*"-")
    print("Avg. sentences per blurb: ", statistics.mean(sentences_per_blurb))
    print("Std. deviation : ", statistics.stdev(sentences_per_blurb))
    print()
    print("Avg. tokens per blurb: ", statistics.mean(tokens_per_blurb))
    print("Std. deviation : ", statistics.stdev(tokens_per_blurb))
    print("\n")
    sentences_per_blurb, tokens_per_blurb = count_sentences_tokens(test_data_x)
    print("Test data")
    print(len("Test data") * "-")
    print("Avg. sentences per blurb: ", statistics.mean(sentences_per_blurb))
    print("Std. deviation : ", statistics.stdev(sentences_per_blurb))
    print()
    print("Avg. tokens per blurb: ", statistics.mean(tokens_per_blurb))
    print("Std. deviation : ", statistics.stdev(tokens_per_blurb))

    # Avg. genres per blurb
    print()
    print("Genres")
    print("======")
    stats_genres_per_blurb(train_data_y)

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
    # load train and test data
    train_data_x, train_data_y, labels = load_data('blurbs_train_all.txt')

    # load dev data
    dev_data_x, dev_data_y, labels = load_data('blurbs_train_all.txt')

    # data analysis
    data_analysis(train_data_x, train_data_y, test_data_x)


if __name__ == '__main__':
    main()
