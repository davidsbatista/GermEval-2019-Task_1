#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from utils import load_data


def data_analysis(train_data_x, train_data_y, labels):

    # TODO: top-words per class
    author_topic = defaultdict(int)

    # for sample_x, sample_y in zip(train_data_x, train_data_y):
    #     # new_data_x = [x['title'] + " SEP " + x['body'] for x in train_data_x]
    #     print(len(sample_x['title'].split()))
    #     print(len(sample_x['body'].split()))
    #     print(len(sample_y))
    #     print()

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
