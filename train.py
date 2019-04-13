#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from collections import defaultdict
from copy import deepcopy

from gensim.models import KeyedVectors

from keras.layers import Embedding, np
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

from analysis import extract_hierarchy
from models.convnets_utils import get_cnn_multichannel, get_cnn_rand, get_cnn_pre_trained_embeddings
from models.keras_han.model import HAN
from models.neural_networks_keras import build_lstm_based_model, build_token_index, \
    vectorize_dev_data, vectorizer
from utils import generate_submission_file, load_data


def train_random_forest(train_x, train_y, test_x, test_y, ml_binarizer, level=None):
    stop_words = set(stopwords.words('german'))
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 2), max_df=0.75)),
        ('clf', RandomForestClassifier())
    ])

    parameters = {
        "clf__min_samples_split": [10, 100, 1000],
        "clf__n_estimators": [250, 300, 500],
        "clf__class_weight": ['balanced', None]
    }

    """
    class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=10,
                           min_weight_fraction_leaf=0.0, n_estimators=250, n_jobs=None,
                           oob_score=False, random_state=None, verbose=0,
                           warm_start=False))
    """

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
        "clf__estimator__C": [300]
    }
    grid_search_tune = GridSearchCV(pipeline, parameters, cv=3, n_jobs=3, verbose=2)
    grid_search_tune.fit(train_x, train_y)

    # measuring performance on test set
    print("Applying best classifier on test data:")
    best_clf = grid_search_tune.best_estimator_
    predictions_prob = best_clf.predict_proba(test_x)

    predictions_probs = np.where(predictions_prob >= 0.5, 1, 0)

    pred_labels = ml_binarizer.inverse_transform(predictions_probs)
    true_labels = ml_binarizer.inverse_transform(test_y)

    top_missed = defaultdict(int)
    missed = 0
    for pred, true, text, probs in zip(pred_labels, true_labels, test_x, predictions_prob):
        if len(pred) == 0:
            missed += 1
            top_missed[true] += 1
            print(text)
            print(len(text.split()))
            print(true)
            print(ml_binarizer.classes_)
            print(probs)
            print()
            print()

    print("Missing labels for samples")
    for k, v in top_missed.items():
        print(k, v)
    print("total missed: ", missed)

    report = classification_report(test_y, pred_labels, target_names=ml_binarizer.classes_)
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


"""
def keras_grid_search():
    def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
        model = Sequential()
        model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
        model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    param_grid = dict(num_filters=[32, 64, 128],
                      kernel_size=[3, 5, 7],
                      vocab_size=[5000],
                      embedding_dim=[50],
                      maxlen=[100])

    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import RandomizedSearchCV

    # Main settings
    epochs = 20
    embedding_dim = 50
    maxlen = 100
    output_file = 'data/output.txt'

    # Run grid search for each source (yelp, amazon, imdb)
    for source, frame in df.groupby('source'):
        print('Running grid search for data set :', source)
        sentences = df['sentence'].values
        y = df['label'].values

        # Train-test split
        sentences_train, sentences_test, y_train, y_test = train_test_split(
            sentences, y, test_size=0.25, random_state=1000)

        # Tokenize words
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(sentences_train)
        X_train = tokenizer.texts_to_sequences(sentences_train)
        X_test = tokenizer.texts_to_sequences(sentences_test)

        # Adding 1 because of reserved 0 index
        vocab_size = len(tokenizer.word_index) + 1

        # Pad sequences with zeros
        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

        # Parameter grid for grid search
        param_grid = dict(num_filters=[32, 64, 128],
                          kernel_size=[3, 5, 7],
                          vocab_size=[vocab_size],
                          embedding_dim=[embedding_dim],
                          maxlen=[maxlen])
        model = KerasClassifier(build_fn=create_model,
                                epochs=epochs, batch_size=10,
                                verbose=False)
        grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                                  cv=4, verbose=1, n_iter=5)
        grid_result = grid.fit(X_train, y_train)

        # Evaluate testing set
        test_accuracy = grid.score(X_test, y_test)

        # Save and evaluate results
        prompt = input(f'finished {source}; write to file and proceed? [y/n]')
        if prompt.lower() not in {'y', 'true', 'yes'}:
            break
        with open(output_file, 'a') as f:
            s = ('Running {} data set\nBest Accuracy : '
                 '{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')
            output_string = s.format(
                source,
                grid_result.best_score_,
                grid_result.best_params_,
                test_accuracy)
            print(output_string)
            f.write(output_string)
"""


def train_han(train_data_x, train_data_y):
    token2idx, max_sent_len = build_token_index(train_data_x)

    # y_data: encode into one-hot vectors
    ml_binarizer = MultiLabelBinarizer()
    y_labels = ml_binarizer.fit_transform(train_data_y)
    print('Total of {} classes'.format(len(ml_binarizer.classes_)))

    # Construct the input matrix. This should be a nd-array of
    # shape (n_samples, MAX_SENT, MAX_WORDS_PER_SENT).
    # We zero-pad this matrix (this does not influence
    # any predictions due to the attention mechanism.

    max_sent = 0
    max_tokens = 0
    for x in train_data_x:
        sentences = sent_tokenize(x['body'], language='german')
        if len(sentences) > max_sent:
            max_sent = len(sentences)
        for sentence in sentences:
            tokens = word_tokenize(sentence, language='german')
            if len(tokens) > max_tokens:
                max_tokens = len(tokens)

    print(max_sent)
    print(max_tokens)

    processed_x = np.zeros((len(train_data_x), max_sent, max_tokens), dtype='int32')

    for i, x in enumerate(train_data_x):
        vectorized_sentences = []
        text = x['title'] + " . " + x['body']
        sentences = sent_tokenize(text, language='german')
        for s in sentences:
            vectorized_sentences.append(vectorizer(word_tokenize(s, language='german')))

        padded_sentences = pad_sequences(vectorized_sentences, padding='post',
                                         truncating='post', maxlen=max_tokens,
                                         value=token2idx['PADDED'])

        pad_size = max_sent - padded_sentences.shape[0]

        if pad_size < 0:
            padded_sentences = padded_sentences[0:max_sent]
        else:
            padded_sentences = np.pad(padded_sentences, ((0, pad_size), (0, 0)), mode='constant',
                                      constant_values=0)

        # Store this observation as the i-th observation in the data matrix
        processed_x[i] = padded_sentences[None, ...]

    print(processed_x.shape)

    # split into train and hold out set
    train_x, test_x, train_y, test_y = train_test_split(processed_x, y_labels,
                                                        random_state=42,
                                                        test_size=0.30)
    print(train_x.shape)
    print(train_y.shape)

    MAX_WORDS_PER_SENT = max_tokens
    MAX_SENT = max_sent
    MAX_VOC_SIZE = 20000
    GLOVE_DIM = 100

    print("Loading pre-trained Embeddings\n")
    static_embeddings = KeyedVectors.load('resources/de-wiki-fasttext-300d-1M')
    # build a word embeddings matrix, out-of-vocabulary words will be initialized randomly
    embedding_matrix = np.random.random((len(token2idx), static_embeddings.vector_size))
    not_found = 0
    for word, i in token2idx.items():
        try:
            embedding_vector = static_embeddings[word.lower()]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            not_found += 1

    print(embedding_matrix)

    han_model = HAN(MAX_WORDS_PER_SENT, MAX_SENT, 8, embedding_matrix,
                    word_encoding_dim=100, sentence_encoding_dim=100)

    han_model.summary()
    han_model.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['acc'])

    han_model.fit(train_x, train_y, batch_size=16, epochs=10, validation_split=0.2)

    predictions = han_model.predict(test_x, verbose=1)

    # ToDo: there must be a more efficient way to do this, BucketEstimator
    binary_predictions = []
    for pred in predictions:
        binary_predictions.append([0 if i <= 0.5 else 1 for i in pred])
    print(classification_report(test_y, np.array(binary_predictions),
                                target_names=ml_binarizer.classes_))

    return han_model, ml_binarizer, max_sent_len, token2idx


def train_cnn_sent_class(train_data_x, train_data_y):

    token2idx, max_sent_len = build_token_index(train_data_x)

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

    # y_data: encode into one-hot vectors
    ml_binarizer = MultiLabelBinarizer()
    y_labels = ml_binarizer.fit_transform(train_data_y)
    print('Total of {} classes'.format(len(ml_binarizer.classes_)))
    n_classes = len(ml_binarizer.classes_)
    train_data_x = vectors_padded
    data_y = y_labels

    # ToDo: do a proper cv validation
    # split into train and hold out set
    train_x, test_x, train_y, test_y = train_test_split(train_data_x, data_y,
                                                        random_state=42,
                                                        test_size=0.25)
    print(train_x.shape)
    print(train_y.shape)

    model = get_cnn_rand(300, len(token2idx) + 1, max_sent_len, n_classes)
    model.fit(train_x, train_y, batch_size=32, epochs=1, verbose=True, validation_split=0.33)
    predictions = model.predict([test_x], verbose=1)

    """
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
    model = get_cnn_pre_trained_embeddings(embedding_layer_static, max_sent_len, n_classes)
    model.fit(train_x, train_y, batch_size=32, epochs=1, verbose=True, validation_split=0.33)
    predictions = model.predict([test_x], verbose=1)
    """

    """
    embedding_layer_dynamic = Embedding(len(token2idx), static_embeddings.vector_size,
                                        weights=[embedding_matrix], input_length=max_sent_len,
                                        trainable=True, name='embeddings_dynamic')
    embedding_layer_static = Embedding(len(token2idx), static_embeddings.vector_size,
                                       weights=[embedding_matrix], input_length=max_sent_len,
                                       trainable=False, name='embeddings_static')
    model = get_cnn_multichannel(embedding_layer_static, embedding_layer_dynamic, max_sent_len,
                                 n_classes)
    model.fit([train_x, train_x], train_y, batch_size=128, epochs=1, validation_split=0.2)
    predictions = model.predict([test_x, test_x], verbose=1)
    """

    # ToDo: there must be a more efficient way to do this, BucketEstimator
    binary_predictions = []
    for pred in predictions:
        binary_predictions.append([0 if i <= 0.5 else 1 for i in pred])
    report = classification_report(test_y, np.array(binary_predictions),
                                   target_names=ml_binarizer.classes_)
    print(report)

    return model, ml_binarizer, max_sent_len, token2idx


def train_cnn_multilabel(train_data_x, train_data_y):

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

    print("\n\n=== TOP-LEVEL ===")
    print(f'top classifier on {len(hierarchical_level_1.keys())} labels')
    print(f'samples {len(data_y_level_0)}')
    print()
    samples_y = [list(y) for y in data_y_level_0]
    top_clf, ml_binarizer, max_sent_len, token2idx = train_cnn_sent_class(train_data_x, samples_y)
    classifiers['top_level']['clf'] = top_clf
    classifiers['top_level']['binarizer'] = ml_binarizer
    classifiers['top_level']['token2idx'] = token2idx
    classifiers['top_level']['max_sent_len'] = max_sent_len

    print("\n\n=== LEVEL 1 ===")
    for k, v in sorted(hierarchical_level_1.items()):
        if len(v) == 0:
            continue
        print(f'classifier {k} on {len(v)} labels')

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

        print("samples: ", len(samples_x))
        print("samples: ", len(samples_y))

        clf, ml_binarizer, max_sent_len, token2idx = train_cnn_sent_class(samples_x, samples_y)
        classifiers['level_1'][k]['clf'] = clf
        classifiers['level_1'][k]['binarizer'] = ml_binarizer
        classifiers['level_1'][k]['token2idx'] = token2idx
        classifiers['level_1'][k]['max_sent_len'] = max_sent_len
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
        print("samples: ", len(samples_y))

        clf, ml_binarizer, max_sent_len, token2idx = train_cnn_sent_class(samples_x, samples_y)
        classifiers['level_2'][k]['clf'] = clf
        classifiers['level_2'][k]['binarizer'] = ml_binarizer
        classifiers['level_2'][k]['token2idx'] = token2idx
        classifiers['level_2'][k]['max_sent_len'] = max_sent_len

        print("----------------------------")

    return classifiers


def subtask_a(train_data_x, train_data_y, dev_data_x, clf='logit'):
    """
    Subtask A

    :param clf:
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

    if clf == 'logit':
        # TF-IDF w/ logistic regression
        model, ml_binarizer = train_baseline(train_data_x, train_data_y)

        # apply on dev data
        new_data_x = [x['title'] + " SEP " + x['body'] for x in dev_data_x]
        predictions = model.predict(new_data_x)

        # for pred, true in zip(predictions, dev_data_x):
        #     print(pred)
        #     all_zeros = not np.any(pred)
        #     if all_zeros:
        #         print(true)

        with open('answer.txt', 'wt') as f_out:
            f_out.write(str('subtask_a\n'))
            for pred, data in zip(ml_binarizer.inverse_transform(predictions), dev_data_x):
                f_out.write(data['isbn'] + '\t' + '\t'.join([p for p in pred]) + '\n')
    else:
        if clf == 'han':
            model, ml_binarizer, max_sent_len, token2idx = train_han(train_data_x, train_data_y)

        if clf == 'lstm':
            model, ml_binarizer, max_sent_len, token2idx = train_bi_lstm(train_data_x, train_data_y)
            test_vectors = vectorize_dev_data(dev_data_x, max_sent_len, token2idx)
            predictions = model.predict(test_vectors)

        if clf == 'cnn':
            model, ml_binarizer, max_sent_len, token2idx = train_cnn_sent_class(train_data_x,
                                                                                train_data_y)
            test_vectors = vectorize_dev_data(dev_data_x, max_sent_len, token2idx)
            predictions = model.predict([test_vectors, test_vectors])

        binary_predictions = []
        for pred, true in zip(predictions, dev_data_x):
            binary = [0 if i <= 0.5 else 1 for i in pred]
            binary_predictions.append(binary)

        generate_submission_file(np.array(binary_predictions), ml_binarizer, dev_data_x)


def subtask_b(train_data_x, train_data_y, dev_data_x, clf='tree'):
    """
    Subtask B
    =========
    The second task is a hierarchical multi-label classification into multiple writing genres.
    In addition to the very general writing genres additional genres of different specificity can
    be assigned to a book. In total, there are 343 different classes that are hierarchically
    structured.
    """

    if clf == 'tree':

        # sub-task B Train 3 classifiers, one for each level, random forests
        classifiers, ml_binarizers = train_random_forests_multilabel(train_data_x, train_data_y)

        with open('results/models_3_labels.pkl', 'wb') as f_in:
            pickle.dump(classifiers, f_in)

        with open('results/ml_binarizers_3_labels.pkl', 'wb') as f_in:
            pickle.dump(ml_binarizers, f_in)

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
        with open('answer_b.txt', 'wt') as f_in:
            """
            f_out.write(str('subtask_a\n'))
            for x in dev_data_x:
                isbn = x['isbn']
                f_out.write(isbn + '\t' + classification[isbn][0] + '\n')
            """

            f_in.write(str('subtask_b\n'))
            for x in dev_data_x:
                isbn = x['isbn']
                f_in.write(
                    isbn + '\t' + classification[isbn][0] + '\t' + classification[isbn][1] + '\t' +
                    classification[isbn][2] + '\n')

    elif clf == 'cnn':

        """
        # sub-task B Train 3 classifiers, one for each level, random forests
        classifiers = train_cnn_multilabel(train_data_x, train_data_y)
        with open('results/classifiers.pkl', 'wb') as f_out:
            pickle.dump(classifiers, f_out)
        """

        with open('results/classifiers.pkl', 'ob') as f_in:
            classifiers = pickle.load(f_in)

        for k, v in classifiers.items():
            print()
            print(k, v)
            print("")

        exit(-1)

        # apply on dev data
        levels = {0: defaultdict(list),
                  1: defaultdict(list),
                  2: defaultdict(list)}
        classification = {}
        for data in dev_data_x:
            classification[data['isbn']] = deepcopy(levels)

        new_data_x = [x['title'] + " SEP " + x['body'] for x in dev_data_x]

        """
        level = 0
        for clf_level, ml_binarizer in zip(classifiers, ml_binarizers):
            predictions = clf_level.predict(new_data_x)

            for pred, data in zip(ml_binarizer.inverse_transform(predictions), dev_data_x):
                classification[data['isbn']][level] = '\t'.join([p for p in pred])
            level += 1
        """

        with open('answer_b.txt', 'wt') as f_in:
            """
            f_out.write(str('subtask_a\n'))
            for x in dev_data_x:
                isbn = x['isbn']
                f_out.write(isbn + '\t' + classification[isbn][0] + '\n')
            """

            f_in.write(str('subtask_b\n'))
            for x in dev_data_x:
                isbn = x['isbn']
                f_in.write(
                    isbn + '\t' + classification[isbn][0] + '\t' + classification[isbn][1] + '\t' +
                    classification[isbn][2] + '\n')


def main():
    # subtask_a
    # subtask_b

    # ToDo: generate a run for a and b by exploring
    #       the hierarchical structure and enforce it in the classifiers

    # ToDo: Naive Bayes para low samples?
    # ToDo: ver os que nao foram atribuidos nenhuma label, forcar tags com base nas palavras ?
    # ToDo: confusion-matrix ?
    # ToDo: grid-search Keras:
    """
    - Grid search across different kernel sizes to find the optimal configuration for your problem,
    in the range 1-10.

    - Search the number of filters from 100-600 and explore a dropout of 0.0-0.5 as part of the
    same search.

    - Explore using tanh, relu, and linear activation functions.
    :return:
    """
    # https://realpython.com/python-keras-text-classification/#convolutional-neural-networks-cnn

    # ToDo: other embeddings? BRET, ELMo, Flair?
    # ToDo: language model based on char?
    # ToDO: https://github.com/cambridgeltl/multilabel-nn
    # ToDo: https://github.com/SarthakMehta/CNN-HAN-for-document-classification
    # ToDo: https://github.com/locuslab/TCN

    # load train data
    train_data_x, train_data_y, labels = load_data('blurbs_train.txt')

    # load dev data
    dev_data_x, _, _ = load_data('blurbs_dev_participants.txt')

    # train subtask_a
    # subtask_a(train_data_x, train_data_y, dev_data_x, clf='han')

    # train subtask_b
    subtask_b(train_data_x, train_data_y, dev_data_x, clf='cnn')


if __name__ == '__main__':
    main()
