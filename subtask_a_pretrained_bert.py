#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random as rn
import pickle

import torch
import numpy as np
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler, WeightedRandomSampler)

import nltk.tokenize
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from models import pretrained_bert
from utils.pre_processing import generate_submission_file, load_data
from utils.pre_processing import vectorize_dev_data, vectorizer

# necessary for starting Numpy generated random numbers in a well-defined initial state.
np.random.seed(42)

# necessary for starting core Python generated random numbers in a well-defined state.
rn.seed(12345)


os.environ['SAMPLING_MINMAX_RATIO'] = os.environ.get('SAMPLING_MINMAX_RATIO', '3')
os.environ['LOSS_MINMAX_RATIO'] = os.environ.get('LOSS_MINMAX_RATIO', '1')
os.environ['NUM_EPOCHS'] = os.environ.get('NUM_EPOCHS', '1')
os.environ['LOSS'] = os.environ.get('LOSS', 'bce')


def get_label_hierarchy():
    """Encode label hierarchy for BERT."""
    labels = []
    with open('./original_label_hierarchy.txt', 'r') as fh:
        for row in fh:
            labels.append(row.strip().split('\t'))
    parents = {}
    hierarchy = {}
    for parent, child in labels:
        parents[child] = parent
        hierarchy[parent] = hierarchy.get(parent, []) + [child]
        hierarchy[child] = []
    label_hierarchy = {}
    label_hierarchy['0'] = [k for k in hierarchy.keys() if k not in parents]
    for i_lbl, lbl in enumerate(label_hierarchy['0']):
        for i, child in enumerate(hierarchy[lbl]):
            label_hierarchy[f'1 - {i_lbl}'] = label_hierarchy.get(f'1 - {i_lbl}', []) + [child]
            for child_ in hierarchy[child]:
                label_hierarchy[f'2 - {i_lbl}:{i}'] = label_hierarchy.get(f'2 - {i_lbl}:{i}', []) + [child_]
    return label_hierarchy, parents


def evaluate_after_epoch(mdl, i_epoch, dev):
    dev_data_x, _, ml_binarizer = dev
    probs = mdl.predict(dev_data_x)
    binary_predictions = np.asarray([row for batch in probs for row in batch.cpu().numpy() > 0.4])
    generate_submission_file(np.array(binary_predictions),
                             ml_binarizer,
                             dev_data_x, suffix=f'-EPOCH_{i_epoch:02d}')
    torch.save(mdl, f'./bert-EPOCH{i_epoch:02d}.pt')

    x_dev, y_dev, *_ = dev
    tokens, masks = mdl.tokenize(x_dev)
    tokens = torch.LongTensor(tokens)
    masks = torch.LongTensor(masks)
    y = torch.FloatTensor(y_dev)
    dev_data = TensorDataset(tokens, y, masks)
    dev_dataloader = DataLoader(dev_data, sampler=SequentialSampler(dev_data),
                                batch_size=mdl.batch_size)

    test_loss = 0
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdl.model.to(DEVICE)
    with torch.no_grad():
        for batch in dev_dataloader:
            batch = (b.to(DEVICE) for b in batch)
            batch_input, batch_targets, batch_masks = batch
            loss, *_ = mdl.model(batch_input,
                                 label_hierarhcy=mdl.label_hierarchy,
                                 labels=batch_targets,
                                 attention_mask=batch_masks,
                                 class_weights=None)
        test_loss += loss.item()
    with open('loss.txt', 'a') as fh:
        fh.write(f'epoch\t{i_epoch}\t{test_loss:.10f}\ttest\n')


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
                                                  loss='bce')
        bert_mdl.post_epoch_hook = evaluate_after_epoch
        try:
            train, test = train_test_split(list(range(len(train_data_x))), test_size=0.25, random_state=89734)
            _data_x, _data_y = [train_data_x[idx] for idx in test], [y_labels[idx] for idx in test]
            _data_x_trn, _data_y_trn = [train_data_x[idx] for idx in train], [y_labels[idx] for idx in train]
            bert_mdl.fit(_data_x_trn, _data_y_trn, dev=(_data_x, _data_y, ml_binarizer))
        finally:
            torch.save(bert_mdl, './pretrained-bert-TASK_A-bce-LVL0.pt')

    bert_mdl = torch.load('./pretrained-bert-TASK_A-bce-LVL0.pt')
    probs = bert_mdl.predict(dev_data_x)
    binary_predictions = np.asarray([row for batch in probs for row in batch.cpu().numpy() > 0.4])
    generate_submission_file(np.array(binary_predictions), ml_binarizer, dev_data_x)


def subtask_b_one_per_level(train_data_x, train_data_y, dev_data_x, train=True):
    label_hierarchy, parents = get_label_hierarchy()
    for lvl, items in label_hierarchy.items():
        label_hierarchy[lvl] = items + ['[N/A]']
    data_y = []
    for y_labels in train_data_y:
        targets = set()
        for label in y_labels:
            # each level should have an extra NOT_APPLICABLE label to allow the loss function to
            # propagate the loss to that label in case nothing from that level was assigned to an
            # instance
            label = {**{0: '[N/A]', 1: '[N/A]', 2: '[N/A]'}, **label}
            targets.update(label.values())
        data_y.append(list(targets))
    train_data_y = data_y

    ml_binarizer = MultiLabelBinarizer()
    y_labels = ml_binarizer.fit_transform(train_data_y).astype(np.float32)
    label_hierarchy = {k: np.where(ml_binarizer.transform([list(v)]).flatten())[0]for k, v in label_hierarchy.items()}

    num_epochs = int(os.environ['NUM_EPOCHS'])
    loss = os.environ['LOSS']
    bert_mdl = pretrained_bert.PretrainedBert(n_epochs=num_epochs,
                                              hierarchical=True,
                                              label_hierarchy=label_hierarchy,
                                              gradient_accumulation_steps=10,
                                              batch_size=10,
                                              loss=loss)
    if train:
        bert_mdl.post_epoch_hook = evaluate_after_epoch
        try:
            train, test = train_test_split(list(range(len(train_data_x))), test_size=0.2, random_state=89734)
            _data_x, _data_y = [train_data_x[idx] for idx in test], [y_labels[idx] for idx in test]
            _data_x_trn, _data_y_trn = [train_data_x[idx] for idx in train], [y_labels[idx] for idx in train]
            bert_mdl.fit(_data_x_trn, _data_y_trn, dev=(_data_x, _data_y, ml_binarizer))
        finally:
            torch.save(bert_mdl, './bert-TASK_B-bce_ONE_PER_LEVEL.pt')

    torch.save(bert_mdl, './bert-TASK_B-bce_ONE_PER_LEVEL.pt')

    bert_mdl = torch.load('./bert-TASK_B-bce_ONE_PER_LEVEL.pt')
    probs = bert_mdl.predict(dev_data_x)
    pred = np.asarray([row for batch in probs for row in batch.cpu().numpy() > 0.4])

    # for each column group (classifier head), set all predictions to False IF the NOT_APPLICABLE
    # label was predicted
    na_idx = np.where(ml_binarizer.transform([['[N/A]']]))[1]
    for level_idx, col_idx in label_hierarchy.items():
        if level_idx == 0 or na_idx not in col_idx:
            continue
        na_predicted = pred[:, col_idx][:, col_idx == na_idx]
        na_predicted = np.where(na_predicted)[0].reshape(-1, 1)
        col_idx_ = [idx for idx in col_idx if idx != na_idx]
        pred[na_predicted, [col_idx_] * na_predicted.shape[0]] = False

    # drop the N/A column from the predictions before passing them to the evaluate script
    not_na_idx = [i for i, cl in enumerate(ml_binarizer.classes_) if cl != '[N/A]']
    final_predictions = pred.copy()[:, not_na_idx]
    ml_binarizer_ = clone(ml_binarizer)
    ml_binarizer_.classes_ = np.asarray([cl for cl in ml_binarizer.classes_ if cl != '[N/A]'])
    generate_submission_file(np.array(final_predictions), ml_binarizer_, dev_data_x, suffix='-bert-bce-B_ONE_PER_LEVEL')

    # enforce the label hierarchy for the predictions:
    # if the parent label was not predicted, ignore the child prediction as well
    label_hierarchy, parents = get_label_hierarchy()
    for child, parent in parents.items():
        child_idx = np.where(ml_binarizer.transform([[child]]))[1]
        if len(child_idx) == 0:  # this child was never seen during training
            continue
        parent_idx = np.where(ml_binarizer.transform([[parent]]))[1]
        parent_not_predicted = pred[:, parent_idx] == False
        pred[parent_not_predicted.flatten(), child_idx] = False

    not_na_idx = [i for i, cl in enumerate(ml_binarizer.classes_) if cl != '[N/A]']
    final_predictions = pred.copy()[:, not_na_idx]
    ml_binarizer_ = clone(ml_binarizer)
    ml_binarizer_.classes_ = np.asarray([cl for cl in ml_binarizer.classes_ if cl != '[N/A]'])
    generate_submission_file(np.array(final_predictions), ml_binarizer_, dev_data_x, suffix='-bert-bce-B_ONE_PER_LEVEL')


def subtask_b_one_per_parent(train_data_x, train_data_y, dev_data_x, train=True):
    label_hierarchy, parents = get_label_hierarchy()
    for lvl, items in label_hierarchy.items():
        label_hierarchy[lvl] = items + ['[N/A]']

    data_y = []
    for y_labels in train_data_y:
        targets = set()
        for label in y_labels:
            # each level should have an extra NOT_APPLICABLE label to allow the loss function to
            # propagate the loss to that label in case nothing from that level was assigned to an
            # instance
            label = {**{0: '[N/A]', 1: '[N/A]', 2: '[N/A]'}, **label}
            targets.update(label.values())
        data_y.append(list(targets))
    train_data_y = data_y

    ml_binarizer = MultiLabelBinarizer()
    y_labels = ml_binarizer.fit_transform(train_data_y).astype(np.float32)
    label_hierarchy = {k: np.where(ml_binarizer.transform([list(v)]).flatten())[0] for k, v in label_hierarchy.items()}

    num_epochs = int(os.environ['NUM_EPOCHS'])
    loss = os.environ['LOSS']
    bert_mdl = pretrained_bert.PretrainedBert(n_epochs=num_epochs,
                                              hierarchical=True,
                                              label_hierarchy=label_hierarchy,
                                              gradient_accumulation_steps=10,
                                              batch_size=10,
                                              loss=loss)
    if train:
        bert_mdl.post_epoch_hook = evaluate_after_epoch
        try:
            train, test = train_test_split(list(range(len(train_data_x))), test_size=0.2, random_state=89734)
            _data_x, _data_y = [train_data_x[idx] for idx in test], [y_labels[idx] for idx in test]
            _data_x_trn, _data_y_trn = [train_data_x[idx] for idx in train], [y_labels[idx] for idx in train]
            bert_mdl.fit(_data_x_trn, _data_y_trn, dev=(_data_x, _data_y, ml_binarizer))
        finally:
            torch.save(bert_mdl, './bert-TASK_B-bce-mdl_per_parent.pt')

    bert_mdl = torch.load('./bert-TASK_B-bce-mdl_per_parent.pt')
    probs = bert_mdl.predict(dev_data_x)
    pred = np.asarray([row for batch in probs for row in batch.cpu().numpy() > 0.4])

    # for each column group (classifier head), set all predictions to False IF the NOT_APPLICABLE
    # label was predicted
    na_idx = np.where(ml_binarizer.transform([['[N/A]']]))[1]
    for level_idx, col_idx in label_hierarchy.items():
        if level_idx == 0 or na_idx not in col_idx:
            continue
        na_predicted = pred[:, col_idx][:, col_idx == na_idx]
        na_predicted = np.where(na_predicted)[0].reshape(-1, 1)
        col_idx_ = [idx for idx in col_idx if idx != na_idx]
        pred[na_predicted, [col_idx_] * na_predicted.shape[0]] = False

    # drop the N/A column from the predictions before passing them to the evaluate script
    not_na_idx = [i for i, cl in enumerate(ml_binarizer.classes_) if cl != '[N/A]']
    final_predictions = pred.copy()[:, not_na_idx]
    ml_binarizer_ = clone(ml_binarizer)
    ml_binarizer_.classes_ = np.asarray([cl for cl in ml_binarizer.classes_ if cl != '[N/A]'])
    generate_submission_file(np.array(final_predictions), ml_binarizer_, dev_data_x, suffix='-bert-bce-B_ONE_PER_LEVEL')

    # enforce the label hierarchy for the predictions:
    # if the parent label was not predicted, ignore the child prediction as well
    label_hierarchy, parents = get_label_hierarchy()
    for child, parent in parents.items():
        child_idx = np.where(ml_binarizer.transform([[child]]))[1]
        if len(child_idx) == 0:  # this child was never seen during training
            continue
        parent_idx = np.where(ml_binarizer.transform([[parent]]))[1]
        parent_not_predicted = pred[:, parent_idx] == False
        pred[parent_not_predicted.flatten(), child_idx] = False

    not_na_idx = [i for i, cl in enumerate(ml_binarizer.classes_) if cl != '[N/A]']
    final_predictions = pred.copy()[:, not_na_idx]
    ml_binarizer_ = clone(ml_binarizer)
    ml_binarizer_.classes_ = np.asarray([cl for cl in ml_binarizer.classes_ if cl != '[N/A]'])
    generate_submission_file(np.array(final_predictions), ml_binarizer_, dev_data_x, suffix='-bert-bce-B_ONE_PER_LEVEL')


def main():
    # load dev/train data
    train_data_x, train_data_y, labels = load_data('blurbs_train.txt', dev=True)

    # load dev/test data
    dev_data_x, _, _ = load_data('blurbs_dev_participants.txt', dev=True)

    # train subtask_a
    # subtask_a(train_data_x, train_data_y, dev_data_x, train=os.environ.get('TRAIN', True))
    if os.environ['MODEL'] == 'one_per_level':
        subtask_b_one_per_level(train_data_x, train_data_y, dev_data_x, train=os.environ.get('TRAIN', True))
    if os.environ['MODEL'] == 'one_per_parent':
        subtask_b_one_per_parent(train_data_x, train_data_y, dev_data_x, train=os.environ.get('TRAIN', True))
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
