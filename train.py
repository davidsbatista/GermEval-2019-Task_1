#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from collections import defaultdict
from copy import deepcopy

import nltk
from gensim.models import KeyedVectors
import numpy as np
from keras.layers import Embedding

from keras_preprocessing.sequence import pad_sequences
from nltk import wordpunct_tokenize

from nltk.corpus import stopwords
from nltk.stem.snowball import GermanStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from data_analysis import extract_hierarchy
from models.utils2 import write_reports_to_disk
from utils import generate_submission_file, load_data

from models.convnets_utils import get_cnn_multichannel, get_cnn_rand
from models.convnets_utils import get_cnn_pre_trained_embeddings, get_embeddings_layer
from models.keras_han.model import HAN
from models.utils import build_lstm_based_model, build_token_index
from models.utils import vectorize_dev_data, vectorizer, vectorize_one_sample
from models.bag_of_tricks import BagOfTricks


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
        max_sent = max(max_sent, len(sentences))
        for sentence in sentences:
            tokens = word_tokenize(sentence, language='german')
            max_tokens = max(max_tokens, len(tokens))

    print(max_tokens)
    print(max_sent)

    processed_x = np.zeros((len(train_data_x), max_sent, max_tokens), dtype='int32')

    for i, x in enumerate(train_data_x):
        vectorized_sentences = []
        text = x['title'] + " . " + x['body']
        sentences = sent_tokenize(text, language='german')
        for s in sentences:
            vectorized_sentences.append(vectorizer(word_tokenize(s, language='german'), token2idx))

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
    print("training")
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


def train_bag_of_tricks(train_data_x, train_data_y, level_label):

    bot = BagOfTricks()
    n_top_tokens = 80000

    # build tokens maping and compute freq
    token2idx, max_sent_length, token_freq = build_token_index(train_data_x, lower=True)

    # select only top-k tokens
    print(len(token2idx))
    token2idx = {k: i for i, (k, v) in enumerate(token_freq.most_common(n=n_top_tokens))}
    print(len(token2idx))
    print(max_sent_length)

    bot.token2idx = token2idx
    bot.max_len = max_sent_length

    # map data to vectors of n-grams
    train_data_x = bot.map_data(train_data_x, train_data_y)

    # y_data: encode into one-hot vectors
    ml_binarizer = MultiLabelBinarizer()
    data_y = ml_binarizer.fit_transform(train_data_y)
    print('Total of {} classes'.format(len(ml_binarizer.classes_)))
    n_classes = len(ml_binarizer.classes_)

    # split into train and hold out set
    train_x, test_x, train_y, test_y = train_test_split(train_data_x, data_y,
                                                        random_state=42,
                                                        test_size=0.30)
    print(train_x.shape)
    print(train_y.shape)

    # build a neural network and train a model
    model = bot.build_neural_network(n_classes)
    model.fit(train_x, train_y, batch_size=32, epochs=100, verbose=1)

    predictions = model.predict([test_x], verbose=1)

    binary_predictions = []
    for pred in predictions:
        binary_predictions.append([0 if i <= 0.5 else 1 for i in pred])
    report = classification_report(test_y, np.array(binary_predictions),
                                   target_names=ml_binarizer.classes_)
    print(report)

    with open('classification_report.txt', 'at+') as f_out:
        f_out.write(level_label + '\n')
        f_out.write("=" * len(level_label) + '\n')
        f_out.write(report)
        f_out.write('\n')

    return model, ml_binarizer, max_sent_length, token2idx


def dummy_fun(doc):
    return doc


def train_logit_tf_idf(train_data_x, train_data_y, level_label):
    """

    - TF-IDF weighted vectors as data representation and apply logistic regression with multi-label

    :param level_label:
    :param train_data_x:
    :param train_data_y:
    :return: tuned classifier

    """
    # encode y labels into one-hot vectors
    ml_binarizer = MultiLabelBinarizer()
    y_labels = ml_binarizer.fit_transform(train_data_y)
    print('Total of {} classes'.format(len(ml_binarizer.classes_)))

    data_y = y_labels
    new_data_x = [x['title'] + ". " + x['body'] for x in train_data_x]

    # de_stemmer = GermanStemmer()
    all_doc_tokens = []

    # TODO: remove stop-words?

    for x in new_data_x:
        doc_tokens = []
        for s in sent_tokenize(x, language='german'):
            tokens = wordpunct_tokenize(s)
            words = [w.lower() for w in nltk.Text(tokens) if w.isalpha()]
            doc_tokens.extend(words)
        # doc_tokens_stemmed = [de_stemmer.stem(x) for x in doc_tokens]
        # all_doc_tokens.append(doc_tokens_stemmed)
        all_doc_tokens.append(doc_tokens)

    new_data_x = all_doc_tokens

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
    print()
    print("Best Classifier parameters:")
    print(best_clf)
    print()
    predictions_prob = best_clf.predict_proba(test_x)

    predictions_bins = np.where(predictions_prob >= 0.5, 1, 0)

    pred_labels = ml_binarizer.inverse_transform(predictions_bins)
    true_labels = ml_binarizer.inverse_transform(test_y)

    top_missed = defaultdict(int)
    missed = 0
    for pred, true, text, probs in zip(pred_labels, true_labels, test_x, predictions_prob):
        if len(pred) == 0:
            missed += 1
            top_missed[true] += 1

    print("Missing labels for samples")
    for k, v in top_missed.items():
        print(k, v)
    print("total missed: ", missed)

    report = classification_report(test_y, predictions_bins, target_names=ml_binarizer.classes_)
    print(report)
    with open('classification_report.txt', 'at+') as f_out:
        f_out.write(level_label+'\n')
        f_out.write("="*len(level_label)+'\n')
        f_out.write(report)
        f_out.write('\n')

    # train a classifier on all data using the parameters that yielded best result
    print("Training classifier with best parameters on all data")
    best_tf_idf = grid_search_tune.best_estimator_.steps[0][1]
    clf = grid_search_tune.best_estimator_.steps[1][1]

    best_pipeline = Pipeline([('tfidf', best_tf_idf), ('clf', clf)])
    best_pipeline.fit(new_data_x, data_y)

    with open('test.pkl', 'wb') as f_out:
        pickle.dump(best_pipeline, f_out)

    return best_pipeline, ml_binarizer


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


def train_cnn_sent_class(train_data_x, train_data_y, level_label):
    # ToDo: grid-search Keras:
    """
    - Grid search across different kernel sizes to find the optimal configuration for your problem,
      in the range 1-10.
    - Search the number of filters from 100-600 and explore a dropout of 0.0-0.5 as part of the
      same search.
    - Explore using tanh, relu, and linear activation functions.
    - https://realpython.com/python-keras-text-classification/#convolutional-neural-networks-cnn

    - See function above
    """

    token2idx, max_sent_len, _ = build_token_index(train_data_x)

    # x_data: vectorize, i.e. tokens to indexes and pad
    print("Vectorizing input data\n")
    vectors = []
    for x in train_data_x:
        tokens = []
        text = x['title'] + " SEP " + x['body']
        sentences = sent_tokenize(text, language='german')
        for s in sentences:
            tokens += word_tokenize(s)
        vector = vectorizer(tokens, token2idx)
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

    # split into train and hold out set
    train_x, test_x, train_y, test_y = train_test_split(train_data_x, data_y,
                                                        random_state=42,
                                                        test_size=0.30)

    print(train_x.shape)
    print(train_y.shape)

    """    
    model = get_cnn_rand(200, len(token2idx) + 1, max_sent_len, n_classes)
    model.fit(train_x, train_y, batch_size=32, epochs=5, verbose=True, validation_split=0.33)
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

    embedding_layer = get_embeddings_layer(embedding_matrix, 'static-embeddings',
                                           max_sent_len, trainable=True)
    model = get_cnn_pre_trained_embeddings(embedding_layer, max_sent_len, n_classes)
    model.fit(train_x, train_y, batch_size=16, epochs=20, verbose=True, validation_split=0.33)
    predictions = model.predict([test_x], verbose=1)

    # ToDo: there must be a more efficient way to do this
    binary_predictions = []
    for pred in predictions:
        binary_predictions.append([0 if i <= 0.5 else 1 for i in pred])
    report = classification_report(test_y, np.array(binary_predictions),
                                   target_names=ml_binarizer.classes_)

    print(report)

    with open('classification_report.txt', 'at+') as f_out:
        f_out.write(level_label + '\n')
        f_out.write("=" * len(level_label) + '\n')
        f_out.write(report)
        f_out.write('\n')

    # train on all data without validation split
    embedding_layer = get_embeddings_layer(embedding_matrix, 'static-embeddings',
                                           max_sent_len, trainable=True)
    model = get_cnn_pre_trained_embeddings(embedding_layer, max_sent_len, n_classes)
    model.fit(train_data_x, data_y, batch_size=16, epochs=20, verbose=True)

    return model, ml_binarizer, max_sent_len, token2idx


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
                                                                             samples_y,
                                                                             "top-level")
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
            clf, ml_binarizer, max_sent_len, token2idx = train_cnn_sent_class(samples_x, samples_y, k)
            classifiers['level_2'][k]['clf'] = clf
            classifiers['level_2'][k]['binarizer'] = ml_binarizer
            classifiers['level_2'][k]['token2idx'] = token2idx
            classifiers['level_2'][k]['max_sent_len'] = max_sent_len

        print()
        print(classifiers['level_2'].keys())
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
        model, ml_binarizer = train_logit_tf_idf(train_data_x, train_data_y, 'top_level')

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


def subtask_b(train_data_x, train_data_y, dev_data_x, strategy='one'):
    """
    Subtask B)

    strategies

    one : 1) train a single classifier for the top 8 labels;
          2) for each top-label train a classifier using the top-label child-labels
          3) repeat this process for level 2

    """

    if strategy == 'one':

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

    # train subtask_a
    subtask_a(train_data_x, train_data_y, dev_data_x, clf='logit')

    # train subtask_b
    # subtask_b(train_data_x, train_data_y, dev_data_x, strategy='one')


if __name__ == '__main__':
    main()
