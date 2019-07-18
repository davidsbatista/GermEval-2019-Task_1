#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random as rn

import numpy as np

import nltk.tokenize
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.preprocessing import MultiLabelBinarizer

from models import pretrained_bert
from utils.pre_processing import generate_submission_file, load_data
from utils.pre_processing import vectorize_dev_data, vectorizer

# necessary for starting Numpy generated random numbers in a well-defined initial state.
np.random.seed(42)

# necessary for starting core Python generated random numbers in a well-defined state.
rn.seed(12345)


def subtask_a(train_data_x, train_data_y, dev_data_x):

    data_y_level_0 = []
    for y_labels in train_data_y:
        labels_0 = set()
        for label in y_labels:
            labels_0.add(label[0])
        data_y_level_0.append(list(labels_0))
    train_data_y = data_y_level_0

    ml_binarizer = MultiLabelBinarizer()
    y_labels = ml_binarizer.fit_transform(train_data_y).astype(np.float32)

    bert_mdl = pretrained_bert.PretrainedBert()
    bert_mdl.fit(train_data_x[:500], y_labels[:500])

    import ipdb; ipdb.set_trace()
    binary_predictions = []
    for pred in predictions:
        binary = [0 if i <= 0.4 else 1 for i in pred]
        if np.all(binary == 0):
            binary = [0 if i <= 0.3 else 1 for i in pred]
        binary_predictions.append(binary)

    generate_submission_file(np.array(binary_predictions), ml_binarizer, dev_data_x)


def main():
    # load dev/train data
    train_data_x, train_data_y, labels = load_data('blurbs_train.txt', dev=True)

    # load dev/test data
    dev_data_x, _, _ = load_data('blurbs_dev_participants.txt', dev=True)

    # train subtask_a
    # subtask_a(train_data_x, train_data_y, dev_data_x, clf='bag-of-tricks')
    # model = subtask_a(train_data_x, train_data_y, dev_data_x, clf='han')
    # subtask_a(train_data_x, train_data_y, dev_data_x, clf='lstm')
    # subtask_a(train_data_x, train_data_y, dev_data_x, clf='cnn')
    # subtask_a(train_data_x, train_data_y, dev_data_x, clf='logit')

    # load submission/test data
    train_data_x, train_data_y, labels = load_data('blurbs_train_all.txt', dev=False)
    test_data_x, _, _ = load_data('blurbs_train_all.txt', dev=False)
    subtask_a(train_data_x, train_data_y, test_data_x)


if __name__ == '__main__':
    main()
