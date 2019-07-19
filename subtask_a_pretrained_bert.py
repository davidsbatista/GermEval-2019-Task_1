#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random as rn

import torch
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


def evaluate_after_epoch(mdl, i_epoch, dev):
    dev_data_x, *_ = dev
    probs = mdl.predict(dev_data_x)
    binary_predictions = np.asarray([row for batch in probs for row in batch.cpu().numpy() > 0.4])
    generate_submission_file(np.array(binary_predictions),
                             ml_binarizer,
                             dev_data_x, suffix=f'-EPOCH_{i_epoch:02d}')
    torch.save(mdl.eval().to('cpu'), './pretrained-bert-level-bce-LVL0_EPOCH{i_epoch:02d}.pt')


def subtask_a(train_data_x, train_data_y, dev_data_x, train=True):
    data_y_level_0 = []
    for y_labels in train_data_y:
        labels_0 = set()
        for label in y_labels:
            labels_0.add(label[0])
        data_y_level_0.append(list(labels_0))
    train_data_y = data_y_level_0

    ml_binarizer = MultiLabelBinarizer()
    y_labels = ml_binarizer.fit_transform(train_data_y).astype(np.float32)

    if train:
        bert_mdl = pretrained_bert.PretrainedBert(batch_size=10,
                                                  gradient_accumulation_steps=5,
                                                  n_epochs=10,
                                                  loss='multilabel-softmargin')
        bert_mdl.post_epoch_hook = evaluate_after_epoch 
        try:
            generate
            _data_x, _data_y = 
            bert_mdl.fit(train_data_x, y_labels, dev=(_data_x, _data_y))
        finally:
            torch.save(bert_mdl, './pretrained-bert-level-bce-LVL0.pt')

    bert_mdl = torch.load('./pretrained-bert-level-bce-LVL0.pt')
    probs = bert_mdl.predict(dev_data_x)
    binary_predictions = np.asarray([row for batch in probs for row in batch.cpu().numpy() > 0.4])
    generate_submission_file(np.array(binary_predictions), ml_binarizer, dev_data_x)


def subtask_b_flat(train_data_x, train_data_y, dev_data_x):
    label_hierarchy = {0: set(), 1: set(), 2: set()}
    data_y = []
    for y_labels in train_data_y:
        targets = {lbl for l_ in y_labels for lbl in l_.values()}
        
        for label in y_labels:
            label = {**{0: '[N/A]', 1: '[N/A]', 2: '[N/A]'}, **label}
            for level, lbl in label.items():
                label_hierarchy[level].add(lbl)
        data_y.append(list(targets))
    train_data_y = data_y
    train_data_y.append(['[N/A]'])

    ml_binarizer = MultiLabelBinarizer()
    y_labels = ml_binarizer.fit_transform(train_data_y).astype(np.float32)
    label_hierarchy = {k: np.where(ml_binarizer.transform([list(v)]).flatten())[0]for k, v in label_hierarchy.items()}

    bert_mdl = pretrained_bert.PretrainedBert(hierarchical=True, label_hierarchy=label_hierarchy)
    bert_mdl.fit(train_data_x[:500], y_labels[:500])

    torch.save(bert_mdl, './pretrained-bert-level-0.pt')

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
    subtask_a(train_data_x, train_data_y, dev_data_x, train=True)
    # model = subtask_a(train_data_x, train_data_y, dev_data_x, clf='han')
    # subtask_a(train_data_x, train_data_y, dev_data_x, clf='lstm')
    # subtask_a(train_data_x, train_data_y, dev_data_x, clf='cnn')
    # subtask_a(train_data_x, train_data_y, dev_data_x, clf='logit')

    # load submission/test data
    # train_data_x, train_data_y, labels = load_data('blurbs_train_all.txt', dev=False)
    # test_data_x, _, _ = load_data('blurbs_test_participants.txt', dev=False)
    # subtask_a(train_data_x, train_data_y, test_data_x, no_train=False)


if __name__ == '__main__':
    main()
