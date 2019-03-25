#!/usr/bin/env python
# -*- coding: utf-8 -*-
from gensim.models import KeyedVectors

from keras import Input, Model, optimizers
from keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM, np
from keras_preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from utils import load_data

token2idx = {}
idx2token = {}
PADDED = 1
UNKNOWN = 0
max_sent_length = 0


def build_token_index(x_data):
    """

    :param x_data:
    :return:
    """

    # index of tokens
    global token2idx
    global idx2token
    global max_sent_length

    vocabulary = set()

    for x in x_data:
        tmp_len = 0
        text = x['title'] + " SEP " + x['body']
        sentences = sent_tokenize(text, language='german')
        for s in sentences:
            vocabulary.update(word_tokenize(s))
            tmp_len += len(s)
        max_sent_length = tmp_len if tmp_len > max_sent_length else max_sent_length

    token2idx = {word: i + 2 for i, word in enumerate(vocabulary, 0)}
    token2idx["PADDED"] = PADDED
    token2idx["UNKNOWN"] = UNKNOWN
    idx2token = {value: key for key, value in token2idx.items()}


def vectorizer(x_sample):
    """
    Something like a Vectorizer, that converts your sentences into vectors,
    either one-hot-encodings or embeddings;

    :return:
    """

    unknown_tokens = 0
    vector = []
    for token in x_sample:
        if token in token2idx:
            vector.append(token2idx[token])
        else:
            unknown_tokens += 1
            vector.append(UNKNOWN)

    return vector


def build_lstm_based_model(embeddings, label_encoder):
    """

    """
    hidden_units = 128
    dropout = 0.2
    recurrent_dropout = 0.3
    dense_dropout = 0.1
    learning_rate = 0.001

    # build a word embeddings matrix, out of vocabulary words will be initialized randomly
    embedding_matrix = np.random.random((len(token2idx), embeddings.vector_size))
    not_found = 0
    for word, i in token2idx.items():
        try:
            embedding_vector = embeddings[word.lower()]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            not_found += 1

    # model itself
    embedding_layer = Embedding(len(token2idx), embeddings.vector_size,
                                weights=[embedding_matrix], input_length=max_sent_length,
                                trainable=True, name='embeddings')

    sequence_input = Input(shape=(max_sent_length,), dtype='int32', name='messages')
    embedded_sequences = embedding_layer(sequence_input)
    l_lstm = Bidirectional(LSTM(hidden_units, dropout=dropout,
                                recurrent_dropout=recurrent_dropout))(embedded_sequences)
    l_lstm_w_drop = Dropout(dense_dropout)(l_lstm)
    preds = Dense(len(label_encoder.classes_),
                  activation='softmax', name='softmax')(l_lstm_w_drop)
    model = Model(inputs=[sequence_input], outputs=[preds])

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=learning_rate),
                  metrics=['acc'])

    print('{} out of {} words randomly initialized'.format(not_found, len(token2idx)))

    return model


def train_model_level_0_baseline(train_data_x, train_data_y):
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

    #train_data_x = train_data_x[:1000]
    #data_y = y_labels[:1000]
    data_y = y_labels

    # TODO: use author as feature, what about unseen authors ?

    new_data_x = [x['title'] + "SEP" + x['body'] for x in train_data_x]

    # split into train and hold out set
    train_x, test_x, train_y, test_y = train_test_split(new_data_x,
                                                        data_y,
                                                        random_state=42,
                                                        test_size=0.20)

    stop_words = set(stopwords.words('german'))
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stop_words)),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='sag', max_iter=3000), n_jobs=3))
    ])
    parameters = {
        'tfidf__max_df': (0.25, 0.5, 0.75),
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        "clf__estimator__C": [0.01, 0.1, 1],
        "clf__estimator__class_weight": ['balanced', None],
    }
    grid_search_tune = GridSearchCV(pipeline, parameters, cv=3, n_jobs=3, verbose=2)
    grid_search_tune.fit(train_x, train_y)
    print("Best parameters set:")
    print(grid_search_tune.best_estimator_.steps)

    # measuring performance on test set
    print("Applying best classifier on test data:")
    best_clf = grid_search_tune.best_estimator_
    predictions = best_clf.predict(test_x)
    print(classification_report(test_y, predictions, target_names=ml_binarizer.classes_))

    return best_clf, ml_binarizer


def train_model_level_0_embeddings(train_data_x, train_data_y):
    pass


def main():

    """
    Subtasks
    This shared task consists of two subtask, described below. You can participate in one of
    them or both.

    Subtask A
    =========
    The task is to classify german books into one or multiple most general writing genres (d=0).
    Therfore, it can be considered a multi-label classification task. In total, there are 8 classes
    that can be assigned to a book:
    - Literatur & Unterhaltung,
    - Ratgeber,
    - Kinderbuch & Jugendbuch,
    - Sachbuch,
    - Ganzheitliches Bewusstsein,
    - Glaube & Ethik,
    - Künste,
    - Architektur & Garten.


    Subtask B
    =========
    The second task is a hierarchical multi-label classification into multiple writing genres.
    In addition to the very general writing genres additional genres of different specificity can
    be assigned to a book. In total, there are 343 different classes that are hierarchically
    structured.

    :return:
    """
    # load train data
    train_data_x, train_data_y, labels = load_data('blurbs_train.txt', hierarchical=False)

    # for k, v in labels.items():
    #     print(k)
    #     print(len(v))
    #     print(v)
    #     print()

    # best_clf, ml_binarizer = train_model(train_data_x, train_data_y)
    # print(best_clf)

    # apply on  dev data
    # dev_data_x, _, _ = load_data('blurbs_dev_participants.txt')
    # new_data_x = [x['title'] + "SEP" + x['body'] for x in dev_data_x]
    # predictions = best_clf.predict(new_data_x)
    # generate_submission_file(predictions, ml_binarizer, dev_data_x)

    # embeddings for neural network
    build_token_index(train_data_x)

    # y_data: encode into one-hot vectors
    ml_binarizer = MultiLabelBinarizer()
    y_labels = ml_binarizer.fit_transform(train_data_y)
    print('Total of {} classes'.format(len(ml_binarizer.classes_)))

    print("Vectorizing input data")
    # x_data: vectorize, i.e. tokens to indexes and pad
    vectors = []
    for x in train_data_x:
        tokens = []
        text = x['title'] + " SEP " + x['body']
        sentences = sent_tokenize(text, language='german')
        for s in sentences:
            tokens += word_tokenize(s)
        vector = vectorizer(tokens)
        vectors.append(vector)
    vectors_padded = pad_sequences(vectors, padding='post', maxlen=max_sent_length,
                                   truncating='post', value=token2idx['PADDED'])

    #train_data_x = vectors_padded[:100]
    #data_y = y_labels[:100]
    data_y = y_labels

    # split into train and hold out set
    train_x, test_x, train_y, test_y = train_test_split(train_data_x,
                                                        data_y,
                                                        random_state=42,
                                                        test_size=0.20)

    print("Loading pre-trained Embeddings")
    static_embeddings = KeyedVectors.load('resources/de-wiki-fasttext-300d-1M')
    model = build_lstm_based_model(static_embeddings, ml_binarizer)

    model.fit(train_x, train_y, batch_size=64, epochs=1, verbose=1, validation_split=0.2)

    predictions = model.predict(test_x)

    # ToDo: there must be a more efficient way to do this
    binary_predictions = []
    for pred in predictions:
        binary_predictions.append([0 if i <= 0.5 else 1 for i in pred])

    print(classification_report(test_y, np.array(binary_predictions), target_names=ml_binarizer.classes_))

if __name__ == '__main__':
    main()
