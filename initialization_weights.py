#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from analysis import extract_hierarchy
from utils import load_data


def create_weight_matrix(n_samples):

    hierarchical_level_1, hierarchical_level_2 = extract_hierarchy()

    label_idx = 0
    labels2idx = {}

    for k in sorted(hierarchical_level_1.keys()):
        labels2idx[k] = label_idx
        label_idx += 1

    for k, v in sorted(hierarchical_level_1.items()):
        for sublabel in v:
            labels2idx[sublabel] = label_idx
            label_idx += 1

    for k, v in sorted(hierarchical_level_2.items()):
        for sublabel in v:
            labels2idx[sublabel] = label_idx
            label_idx += 1

    weight_matrix = np.zeros((n_samples, len(labels2idx)))

    return weight_matrix, labels2idx


def init_weight_matrix(matrix, train_data_y, labels2idx):

    for i, y_sample in enumerate(train_data_y):
        for labels in y_sample:
            for level, label in labels.items():
                j = labels2idx[label]
                matrix[i, j] = 1
    return matrix


def main():
    # load train data
    train_data_x, train_data_y, labels = load_data('blurbs_train.txt')

    # create matrix
    weight_matrix, labels2idx = create_weight_matrix(n_samples=14548)
    print(weight_matrix.shape)

    # fill-in weight matrix
    weight_matrix = init_weight_matrix(weight_matrix, train_data_y, labels2idx)

    idx2labels = {v: k for k, v in labels2idx.items()}

    for row, y_sample in zip(weight_matrix, train_data_y):
        non_zeros = np.nonzero(row)
        for x in non_zeros[0]:
            print(idx2labels[x])
        print(y_sample)
        print("----------")

if __name__ == '__main__':
    main()
