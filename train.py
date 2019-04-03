#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from collections import defaultdict
from copy import deepcopy

from gensim.models import KeyedVectors

from keras.layers import np, Embedding
from keras_preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import class_weight, compute_sample_weight

from models.convnets_utils import get_cnn_rand, get_cnn_pre_trained_embeddings, \
    create_embeddings_matrix, get_embeddings_layer, load_fasttext_embeddings, get_cnn_multichannel
from models.neural_networks_keras import build_lstm_based_model, build_token_index, vectorizer
from utils import load_data, generate_submission_file


def train_bi_lstm(train_data_x, train_data_y):
    """
    Trains a biLSTM classifier, message is represented by the concatenation of the two last
    states from each LSTM.

    :param train_data_x:
    :param train_data_y:
    :return:
    """
    token2idx, max_sent_len = build_token_index(train_data_x)

    # y_data: encode into one-hot vectors
    ml_binarizer = MultiLabelBinarizer()
    y_labels = ml_binarizer.fit_transform(train_data_y)
    print('Total of {} classes'.format(len(ml_binarizer.classes_)))

    # x_data: vectorize, i.e. tokens to indexes and pad
    print("Vectorizing input data\n")
    vectors = []
    for x in train_data_x:
        tokens = []
        text = x['title'] + " SEP " + x['body']
        sentences = sent_tokenize(text, language='german')
        for s in sentences:
            tokens += word_tokenize(s)
        vector = vectorizer(tokens)
        vectors.append(vector)
    vectors_padded = pad_sequences(vectors, padding='post', maxlen=max_sent_len,
                                   truncating='post', value=token2idx['PADDED'])

    train_data_x = vectors_padded
    data_y = y_labels

    # split into train and hold out set
    train_x, test_x, train_y, test_y = train_test_split(train_data_x, data_y,
                                                        random_state=42,
                                                        test_size=0.20)
    print("Loading pre-trained Embeddings\n")
    static_embeddings = KeyedVectors.load('resources/de-wiki-fasttext-300d-1M')
    model = build_lstm_based_model(static_embeddings, ml_binarizer, max_sent_len)

    # since we have imbalanced dataset
    # sample_weights = compute_sample_weight('balanced', train_y)
    model.fit(train_x, train_y, batch_size=16, epochs=5, verbose=1, validation_split=0.2)

    predictions = model.predict(test_x)

    # ToDo: there must be a more efficient way to do this
    binary_predictions = []
    for pred in predictions:
        binary_predictions.append([0 if i <= 0.5 else 1 for i in pred])
    print(classification_report(test_y, np.array(binary_predictions),
                                target_names=ml_binarizer.classes_))

    return model, ml_binarizer, max_sent_len, token2idx


def train_baseline(train_data_x, train_data_y):
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

    data_y = y_labels

    # TODO: use author as feature, what about unseen authors ?

    new_data_x = [x['title'] + " SEP " + x['body'] for x in train_data_x]

    # split into train and hold out set
    train_x, test_x, train_y, test_y = train_test_split(new_data_x, data_y,
                                                        random_state=42,
                                                        test_size=0.30)

    stop_words = set(stopwords.words('german'))
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 2), max_df=0.75)),
        ('clf', OneVsRestClassifier(
            LogisticRegression(class_weight='balanced', solver='sag', max_iter=5000),
            n_jobs=3))
    ])
    parameters = {
        "clf__estimator__C": [200, 250, 300, 350]
    }
    grid_search_tune = GridSearchCV(pipeline, parameters, cv=3, n_jobs=3, verbose=2)
    grid_search_tune.fit(train_x, train_y)

    # measuring performance on test set
    print("Applying best classifier on test data:")
    best_clf = grid_search_tune.best_estimator_
    predictions = best_clf.predict(test_x)
    report = classification_report(test_y, predictions, target_names=ml_binarizer.classes_)
    print(report)
    with open('results/models_subtask_a_report.txt', 'wt') as f_out:
        f_out.write(report)

    # train a classifier on all data using the parameters that yielded best result
    print("Training classifier with best parameters on all data")
    best_tf_idf = grid_search_tune.best_estimator_.steps[0][1]
    clf = grid_search_tune.best_estimator_.steps[1][1]

    print(best_tf_idf)
    print()
    print(clf)

    best_pipeline = Pipeline([('tfidf', best_tf_idf), ('clf', clf)])
    best_pipeline.fit(new_data_x, data_y)

    return best_pipeline, ml_binarizer


def train_random_forest(train_x, train_y, test_x, test_y, ml_binarizer, level=None):
    stop_words = set(stopwords.words('german'))
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 2), max_df=0.75)),
        ('clf', RandomForestClassifier())
    ])

    parameters = {
        # "clf__n_estimators": [10, 100, 1000],
        "clf__n_estimators": [250, 300],
    }
    grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=3, verbose=4)
    grid_search_tune.fit(train_x, train_y)
    print("Best parameters set:")
    print(grid_search_tune.best_estimator_.steps)

    # measuring performance on test set
    # print("Applying best classifier on test data:")
    best_clf = grid_search_tune.best_estimator_
    predictions = best_clf.predict(test_x)
    report = classification_report(test_y, predictions, target_names=ml_binarizer.classes_)
    print(report)
    with open('results/models_subtask_b_report_{}.txt'.format(level), 'wt') as f_out:
        f_out.write(report)

    # train a classifier on all data using the parameters that yielded best result
    print("Training classifier with best parameters on all data")
    best_tf_idf = grid_search_tune.best_estimator_.steps[0][1]
    clf = grid_search_tune.best_estimator_.steps[1][1]
    best_pipeline = Pipeline([('tfidf', best_tf_idf), ('clf', clf)])
    all_data_x = np.concatenate([train_x, test_x])
    all_data_y = np.concatenate([train_y, test_y])
    best_pipeline.fit(all_data_x, all_data_y)

    return best_clf


def train_random_forests_multilabel(train_data_x, train_data_y):
    # aggregate data for 3-independent classifiers
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

    classifiers = []
    ml_binarizers = []

    level = 0
    for train_data_y in [data_y_level_0, data_y_level_1, data_y_level_2]:
        # encode y labels into one-hot vectors
        ml_binarizer = MultiLabelBinarizer()
        y_labels = ml_binarizer.fit_transform(train_data_y)
        print('Total of {} classes'.format(len(ml_binarizer.classes_)))
        data_y = y_labels

        # text representation: merge title and body
        new_data_x = [x['title'] + " SEP " + x['body'] for x in train_data_x]

        # split into train and hold out set
        train_x, test_x, train_y, test_y = train_test_split(new_data_x, data_y,
                                                            random_state=42,
                                                            test_size=0.30)
        clf = train_random_forest(train_x, train_y, test_x, test_y, ml_binarizer, level)
        classifiers.append(clf)
        ml_binarizers.append(ml_binarizer)
        level += 1

    return classifiers, ml_binarizers


def subtask_a(train_data_x, train_data_y, dev_data_x):
    """
    Subtask A
    =========
    The task is to classify german books into one or multiple most general writing genres (d=0).
    Therfore, it can be considered a multi-label classification task. In total, there are 8 classes.
    - Literatur & Unterhaltung,
    - Ratgeber,
    - Kinderbuch & Jugendbuch,
    - Sachbuch,
    - Ganzheitliches Bewusstsein,
    - Glaube & Ethik,
    - Künste,
    - Architektur & Garten.

    :param dev_data_x:
    :param train_data_x:
    :param train_data_y:
    :return:
    """

    data_y_level_0 = []
    for y_labels in train_data_y:
        labels_0 = set()
        for label in y_labels:
            labels_0.add(label[0])
        data_y_level_0.append(list(labels_0))

    train_data_y = data_y_level_0

    # # Subtask-A: Level 0 multi-label classifier
    # model, ml_binarizer = train_baseline(train_data_x, train_data_y)
    #
    # with open('results/models_subtask_a.pkl', 'wb') as f_out:
    #     pickle.dump(model, f_out)
    #
    # with open('results/ml_binarizer_subtask_a.pkl', 'wb') as f_out:
    #     pickle.dump(ml_binarizer, f_out)
    #
    # # apply on dev data
    # new_data_x = [x['title'] + " SEP " + x['body'] for x in dev_data_x]
    # predictions = model.predict(new_data_x)
    #
    # with open('answer.txt', 'wt') as f_out:
    #     f_out.write(str('subtask_a\n'))
    #     for pred, data in zip(ml_binarizer.inverse_transform(predictions), dev_data_x):
    #         f_out.write(data['isbn'] + '\t' + '\t'.join([p for p in pred]) + '\n')

    # # Subtask-A: Neural Networks Approach
    #
    # model, ml_binarizer, max_sent_len, token2idx = train_bi_lstm(train_data_x, train_data_y)
    #
    # print("Vectorizing dev data\n")
    # # dev_data_x: vectorize, i.e. tokens to indexes and pad
    # vectors = []
    # for x in dev_data_x:
    #     tokens = []
    #     text = x['title'] + " SEP " + x['body']
    #     sentences = sent_tokenize(text, language='german')
    #     for s in sentences:
    #         tokens += word_tokenize(s)
    #     vector = vectorizer(tokens)
    #     vectors.append(vector)
    # test_vectors = pad_sequences(vectors, padding='post', maxlen=max_sent_len,
    #                              truncating='post', value=token2idx['PADDED'])
    # predictions = model.predict(test_vectors)
    # binary_predictions = []
    # for pred in predictions:
    #     binary = [0 if i <= 0.5 else 1 for i in pred]
    #     binary_predictions.append(binary)
    # generate_submission_file(np.array(binary_predictions), ml_binarizer, dev_data_x)

    train_cnn_sent_class(train_data_x, train_data_y)


def subtask_b(train_data_x, train_data_y, dev_data_x):
    """
    Subtask B
    =========
    The second task is a hierarchical multi-label classification into multiple writing genres.
    In addition to the very general writing genres additional genres of different specificity can
    be assigned to a book. In total, there are 343 different classes that are hierarchically
    structured.

    """

    # sub-task B Train 3 classifiers, one for each level, random forests
    classifiers, ml_binarizers = train_random_forests_multilabel(train_data_x, train_data_y)

    with open('results/models_3_labels.pkl', 'wb') as f_out:
        pickle.dump(classifiers, f_out)

    with open('results/ml_binarizers_3_labels.pkl', 'wb') as f_out:
        pickle.dump(ml_binarizers, f_out)

    # apply on dev data
    levels = {0: defaultdict(list),
              1: defaultdict(list),
              2: defaultdict(list)}

    classification = {}

    for data in dev_data_x:
        classification[data['isbn']] = deepcopy(levels)

    new_data_x = [x['title'] + " SEP " + x['body'] for x in dev_data_x]
    level = 0
    for clf_level, ml_binarizer in zip(classifiers, ml_binarizers):
        predictions = clf_level.predict(new_data_x)

        for pred, data in zip(ml_binarizer.inverse_transform(predictions), dev_data_x):
            classification[data['isbn']][level] = '\t'.join([p for p in pred])
        level += 1
    with open('answer_b.txt', 'wt') as f_out:
        """
        f_out.write(str('subtask_a\n'))
        for x in dev_data_x:
            isbn = x['isbn']
            f_out.write(isbn + '\t' + classification[isbn][0] + '\n')
        """

        f_out.write(str('subtask_b\n'))
        for x in dev_data_x:
            isbn = x['isbn']
            f_out.write(
                isbn + '\t' + classification[isbn][0] + '\t' + classification[isbn][1] + '\t' +
                classification[isbn][2] + '\n')


def train_cnn_sent_class(train_data_x, train_data_y):

    token2idx, max_sent_len = build_token_index(train_data_x)

    # y_data: encode into one-hot vectors
    ml_binarizer = MultiLabelBinarizer()
    y_labels = ml_binarizer.fit_transform(train_data_y)
    print('Total of {} classes'.format(len(ml_binarizer.classes_)))

    # x_data: vectorize, i.e. tokens to indexes and pad
    print("Vectorizing input data\n")
    vectors = []
    for x in train_data_x:
        tokens = []
        text = x['title'] + " SEP " + x['body']
        sentences = sent_tokenize(text, language='german')
        for s in sentences:
            tokens += word_tokenize(s)
        vector = vectorizer(tokens)
        vectors.append(vector)
    vectors_padded = pad_sequences(vectors, padding='post', maxlen=max_sent_len,
                                   truncating='post', value=token2idx['PADDED'])

    train_data_x = vectors_padded
    data_y = y_labels

    # split into train and hold out set
    train_x, test_x, train_y, test_y = train_test_split(train_data_x, data_y,
                                                        random_state=42,
                                                        test_size=0.30)
    print(train_x.shape)
    print(train_y.shape)

    print("Loading pre-trained Embeddings\n")
    static_embeddings = KeyedVectors.load('resources/de-wiki-fasttext-300d-1M')
    # build a word embeddings matrix, out of vocabulary words will be initialized randomly
    embedding_matrix = np.random.random((len(token2idx), static_embeddings.vector_size))
    not_found = 0
    for word, i in token2idx.items():
        try:
            embedding_vector = static_embeddings[word.lower()]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            not_found += 1

    embedding_layer_dynamic = Embedding(len(token2idx), static_embeddings.vector_size,
                                        weights=[embedding_matrix], input_length=max_sent_len,
                                        trainable=True, name='embeddings_dynamic')

    embedding_layer_static = Embedding(len(token2idx), static_embeddings.vector_size,
                                       weights=[embedding_matrix], input_length=max_sent_len,
                                       trainable=False, name='embeddings_static')

    # model = get_cnn_rand(300, len(token2idx) + 1, max_sent_len, 8)
    # model = get_cnn_pre_trained_embeddings(embedding_layer_static, max_sent_len, 8)
    # model.fit(train_x, train_y, batch_size=32, epochs=10, verbose=True, validation_split=0.2)

    model = get_cnn_multichannel(embedding_layer_static, embedding_layer_dynamic, max_sent_len, 8)
    model.fit([train_x, train_x], train_y, batch_size=32, epochs=10, validation_split=0.2)
    predictions = model.predict([train_x, train_x], verbose=1)

    # ToDo: there must be a more efficient way to do this
    binary_predictions = []
    for pred in predictions:
        binary_predictions.append([0 if i <= 0.5 else 1 for i in pred])
    print(classification_report(test_y, np.array(binary_predictions),
                                target_names=ml_binarizer.classes_))


def main():
    # ToDo: produce a run for subtask-B
    # ToDo: explore the hierarchical structure and enforce it in the classifiers
    # ToDo: ver os que nao foram atribuidos nenhuma label, forcar uma com base nas palavras ?
    # ToDo: confusion-matrix ?

    # load train data
    train_data_x, train_data_y, labels = load_data('blurbs_train.txt')

    # load dev data
    dev_data_x, _, _ = load_data('blurbs_dev_participants.txt')

    # train subtask_a
    subtask_a(train_data_x, train_data_y, dev_data_x)

    # train subtask_b
    # subtask_b(train_data_x, train_data_y, dev_data_x)


if __name__ == '__main__':
    main()
