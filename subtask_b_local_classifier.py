#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from collections import defaultdict
from copy import deepcopy
import numpy as np
from statistical_analysis.data_analysis import extract_hierarchy
from utils.models_utils import train_bag_of_tricks, train_cnn_sent_class, train_logit_tf_idf
from utils.pre_processing import load_data, vectorize_one_sample


def train_clf_per_parent_node(train_data_x, train_data_y, type_clfs):

    # aggregate data for 3-level classifiers
    data_y_level_0 = []
    data_y_level_1 = []
    data_y_level_2 = []

    for y_labels in train_data_y:
        labels_0 = set()
        labels_1 = set()
        labels_2 = set()
        for label in y_labels:
            labels_0.add(label[0])
            if 1 in label:
                labels_1.add(label[1])
            if 2 in label:
                labels_2.add(label[2])
        data_y_level_0.append(labels_0)
        data_y_level_1.append(labels_1)
        data_y_level_2.append(labels_2)

    hierarchical_level_1, hierarchical_level_2 = extract_hierarchy()
    classifiers = {'top_level': defaultdict(dict),
                   'level_1': defaultdict(dict),
                   'level_2': defaultdict(dict)}

    print("type_clfs")
    print(type_clfs)
    print()

    # train a classifier for each level
    print("\n\n=== TOP-LEVEL ===")
    print(f'top classifier on {len(hierarchical_level_1.keys())} labels')
    print(f'samples {len(data_y_level_0)}')
    print()
    samples_y = [list(y) for y in data_y_level_0]

    if type_clfs['top'] == 'logit':
        top_clf, ml_binarizer, = train_logit_tf_idf(train_data_x, samples_y, 'top_level')
        classifiers['top_level']['clf'] = top_clf
        classifiers['top_level']['binarizer'] = ml_binarizer

    elif type_clfs['top'] == 'cnn':
        top_clf, ml_binarizer, max_sent_len, token2idx = train_cnn_sent_class(train_data_x,
                                                                              samples_y,
                                                                              "top-level")
        classifiers['top_level']['clf'] = top_clf
        classifiers['top_level']['binarizer'] = ml_binarizer
        classifiers['top_level']['token2idx'] = token2idx
        classifiers['top_level']['max_sent_len'] = max_sent_len

    elif type_clfs['top'] == 'bag-of-tricks':
        top_clf, ml_binarizer, max_sent_len, token2idx = train_bag_of_tricks(train_data_x,
                                                                             samples_y,)
        classifiers['top_level']['clf'] = top_clf
        classifiers['top_level']['binarizer'] = ml_binarizer
        classifiers['top_level']['token2idx'] = token2idx
        classifiers['top_level']['max_sent_len'] = max_sent_len

    print("\n\n=== LEVEL 1 ===")
    for k, v in sorted(hierarchical_level_1.items()):

        if len(v) == 0:
            continue

        samples_x = [x for x, y in zip(train_data_x, data_y_level_1)
                     if any(label in y for label in v)]
        samples_y = []

        for y in data_y_level_1:
            target = []
            if any(label in y for label in v):
                for label in y:
                    if label in v:
                        target.append(label)
                samples_y.append(target)

        print(f'classifier {k} on {len(v)} labels')
        print("samples: ", len(samples_y))
        print()

        # total number of samples?  samples per classe?
        # ToDo: Naive Bayes para low samples?
        # ToDo: ver os que nao foram atribuidos nenhuma label, forcar tags com base nas palavras ?

        if type_clfs['level_1'] == 'logit':
            clf, ml_binarizer, = train_logit_tf_idf(samples_x, samples_y, k)
            classifiers['level_1'][k]['clf'] = clf
            classifiers['level_1'][k]['binarizer'] = ml_binarizer

        elif type_clfs['level_1'] == 'cnn':
            clf, ml_binarizer, max_sent_len, token2idx = train_cnn_sent_class(samples_x, samples_y,
                                                                              k)
            classifiers['level_1'][k]['clf'] = clf
            classifiers['level_1'][k]['binarizer'] = ml_binarizer
            classifiers['level_1'][k]['token2idx'] = token2idx
            classifiers['level_1'][k]['max_sent_len'] = max_sent_len

        print()
        print(classifiers['level_1'].keys())
        print("----------------------------")

    print("\n\n=== LEVEL 2 ===")
    for k, v in sorted(hierarchical_level_2.items()):
        if len(v) == 0:
            continue
        print(f'classifier {k} on {len(v)} labels')
        samples_x = [x for x, y in zip(train_data_x, data_y_level_2)
                     if any(label in y for label in v)]
        samples_y = []
        for y in data_y_level_2:
            target = []
            if any(label in y for label in v):
                for label in y:
                    if label in v:
                        target.append(label)
                samples_y.append(target)
        print("samples: ", len(samples_x))
        if type_clfs['level_2'] == 'logit':
            clf, ml_binarizer, = train_logit_tf_idf(samples_x, samples_y, k)
            classifiers['level_2'][k]['clf'] = clf
            classifiers['level_2'][k]['binarizer'] = ml_binarizer

        elif type_clfs['level_2'] == 'cnn':
            clf, ml_binarizer, max_sent_len, token2idx = train_cnn_sent_class(samples_x, samples_y,
                                                                              k)
            classifiers['level_2'][k]['clf'] = clf
            classifiers['level_2'][k]['binarizer'] = ml_binarizer
            classifiers['level_2'][k]['token2idx'] = token2idx
            classifiers['level_2'][k]['max_sent_len'] = max_sent_len

        print()
        print(classifiers['level_2'].keys())
        print("----------------------------")

    return classifiers


def subtask_b(train_data_x, train_data_y, dev_data_x):

    out_file = 'results/classifiers.pkl'

    # possibilities: logit, bag-of-tricks, cnn
    clfs = {'top': 'logit',
            'level_1': 'cnn',
            'level_2': 'cnn'}

    classifiers = train_clf_per_parent_node(train_data_x, train_data_y, clfs)

    print(f"Saving trained classifiers to {out_file} ...")
    with open(out_file, 'wb') as f_out:
        pickle.dump(classifiers, f_out)

    print(f"Reading trained classifiers to {out_file} ...")
    with open('results/classifiers.pkl', 'rb') as f_in:
        classifiers = pickle.load(f_in)

    # apply on dev data
    # structure to store predictions on dev_data
    levels = {0: [],
              1: [],
              2: []}
    classification = {}
    for data in dev_data_x:
        classification[data['isbn']] = deepcopy(levels)

    #
    # apply the top-level classifier
    #
    # top_level_clf = classifiers['top_level']['clf']
    # binarizer = classifiers['top_level']['binarizer']
    # token2idx = classifiers['top_level']['token2idx']
    # max_sent_len = classifiers['top_level']['max_sent_len']
    # dev_vector = vectorize_dev_data(dev_data_x, max_sent_len, token2idx)
    # print("Predicting on dev data")
    # predictions = top_level_clf.predict([dev_vector], verbose=1)
    # pred_bin = (predictions > [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).astype(int)
    # for pred, data in zip(binarizer.inverse_transform(pred_bin), dev_data_x):
    #     if pred is None:
    #         continue
    #     classification[data['isbn']][0] = [p for p in pred]
    #     print('\t'.join([p for p in pred]))
    #     print("-----")

    top_level_clf = classifiers['top_level']['clf']
    binarizer = classifiers['top_level']['binarizer']
    print("Predicting on dev data")
    new_data_x = [x['title'] + " SEP " + x['body'] for x in dev_data_x]
    predictions = top_level_clf.predict(new_data_x)
    predictions_bins = np.where(predictions >= 0.5, 1, 0)
    for pred, data in zip(binarizer.inverse_transform(predictions_bins), dev_data_x):
        if pred is None:
            continue
        classification[data['isbn']][0] = [p for p in pred]
        print('\t'.join([p for p in pred]))
        print("-----")

    #
    # apply level-1 classifiers for prediction from the top-level classifier
    #
    for data in dev_data_x:
        top_level_pred = classification[data['isbn']][0]
        if len(top_level_pred) == 0:
            continue
        print("top_level_preds: ", top_level_pred)
        # call level-1 classifier for each pred from top-level
        for pred in top_level_pred:

            # TF-IDF logit
            # clf = classifiers['level_1'][pred]['clf']
            # binarizer = classifiers['level_1'][pred]['binarizer']
            # new_data_x = [x['title'] + " SEP " + x['body'] for x in dev_data_x]
            # print("Predicting on dev data")
            # predictions = clf.predict(new_data_x)
            # pred_bin = np.where(predictions >= 0.5, 1, 0)

            # CNN
            clf = classifiers['level_1'][pred]['clf']
            binarizer = classifiers['level_1'][pred]['binarizer']
            token2idx = classifiers['level_1'][pred]['token2idx']
            max_sent_len = classifiers['level_1'][pred]['max_sent_len']
            dev_vector = vectorize_one_sample(data, max_sent_len, token2idx)
            predictions = clf.predict([dev_vector], verbose=1)
            filtered = np.array(len(binarizer.classes_)*[0.5])
            pred_bin = (predictions > filtered).astype(int)

            indexes = pred_bin[0].nonzero()[0]
            if indexes.any():
                for x in np.nditer(indexes):
                    label = binarizer.classes_[int(x)]
                    print(classification[data['isbn']][1])
                    classification[data['isbn']][1].append(str(label))
                print("\n=====")

    #
    # apply level-2 classifiers for prediction from the top-level classifier
    #
    for data in dev_data_x:
        level_1_pred = classification[data['isbn']][1]
        if len(level_1_pred) == 0:
            continue
        print("level_1_pred: ", level_1_pred)
        for pred in level_1_pred:
            # call level-2 classifier for each pred from top-level
            if pred not in classifiers['level_2']:
                continue

            # clf = classifiers['level_2'][pred]['clf']
            # binarizer = classifiers['level_2'][pred]['binarizer']
            # new_data_x = [x['title'] + " SEP " + x['body'] for x in dev_data_x]
            # print("Predicting on dev data")
            # predictions = clf.predict(new_data_x)
            # pred_bin = np.where(predictions >= 0.5, 1, 0)

            clf = classifiers['level_2'][pred]['clf']
            binarizer = classifiers['level_2'][pred]['binarizer']
            token2idx = classifiers['level_2'][pred]['token2idx']
            max_sent_len = classifiers['level_2'][pred]['max_sent_len']
            dev_vector = vectorize_one_sample(data, max_sent_len, token2idx)
            print("Predicting on dev data")
            predictions = clf.predict([dev_vector], verbose=1)
            filter_threshold = np.array(len(binarizer.classes_)*[0.5])
            pred_bin = (predictions > filter_threshold).astype(int)

            indexes = pred_bin[0].nonzero()[0]
            if indexes.any():
                for x in np.nditer(indexes):
                    label = binarizer.classes_[int(x)]
                    print(classification[data['isbn']][2])
                    classification[data['isbn']][2].append(str(label))
                print("\n=====")

    # generate answer file
    with open('answer.txt', 'wt') as f_out:
        f_out.write(str('subtask_a\n'))
        for x in dev_data_x:
            isbn = x['isbn']
            f_out.write(str(isbn) + '\t' + '\t'.join(classification[isbn][0]) + '\n')

        f_out.write(str('subtask_b\n'))
        for x in dev_data_x:
            isbn = x['isbn']
            output = isbn + '\t'
            output += '\t'.join(classification[isbn][0])
            if len(classification[isbn][1]) > 0:
                output += '\t'
                output += '\t'.join(classification[isbn][1])
                if len(classification[isbn][2]) > 0:
                    output += '\t'
                    output += '\t'.join(classification[isbn][2])
            output += '\n'
            f_out.write(output)


def main():
    # load train data
    train_data_x, train_data_y, labels = load_data('blurbs_train.txt', dev=True)

    # load dev data
    dev_data_x, _, _ = load_data('blurbs_dev_participants.txt', dev=True)

    # train subtask_b
    subtask_b(train_data_x, train_data_y, dev_data_x)


if __name__ == '__main__':
    main()