#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from keras import Input, Model
from keras.layers import Embedding, Convolution1D, GlobalMaxPooling1D, Concatenate, Dense, \
    AlphaDropout
from keras_preprocessing.sequence import pad_sequences
from nltk import sent_tokenize, word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer

from analysis import extract_hierarchy
from models.neural_networks_keras import vectorizer, build_token_index
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


def build_neural_network(weight_matrix, max_input, vocab_size):
    input_size = max_input
    alphabet_size = vocab_size
    embedding_size = 50
    conv_layers = [[256, 10], [256, 7], [256, 5], [256, 3]]
    fully_connected_layers = [weight_matrix.shape[0], weight_matrix.shape[0]]
    dropout_p = 0.1
    num_of_classes = weight_matrix.shape[1]
    optimizer = "adam"
    loss = "categorical_crossentropy"
    threshold = 1e-6

    print("number of units in hidden layer : ", weight_matrix.shape[0])
    print("number of labels in output layer: ", weight_matrix.shape[1])

    # Input layer
    inputs = Input(shape=(input_size,), name='sent_input', dtype='int64')

    # Embedding layers
    x = Embedding(alphabet_size + 1, embedding_size, input_length=input_size)(inputs)

    # Convolution layers
    convolution_output = []
    for num_filters, filter_width in conv_layers:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=filter_width,
                             activation='tanh',
                             name='Conv1D_{}_{}'.format(num_filters, filter_width))(x)
        pool = GlobalMaxPooling1D(
            name='MaxPoolingOverTime_{}_{}'.format(num_filters, filter_width))(conv)
        convolution_output.append(pool)
    x = Concatenate()(convolution_output)

    # Fully connected layers
    for fl in fully_connected_layers:
        x = Dense(fl, activation='selu', kernel_initializer='lecun_normal')(x)
        x = AlphaDropout(dropout_p)(x)

    # Output layer
    predictions = Dense(num_of_classes, activation='softmax')(x)

    # Build and compile model
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=optimizer, loss=loss)

    from keras.utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    return model


def build_vectors(train_data_x, train_data_y, labels2idx):

    # ToDo: vectorize input data and target

    token2idx, max_sent_len = build_token_index(train_data_x)

    print("token2idx: ", len(token2idx))

    # y_data: encode into one-hot vectors with all labels in the hierarchy
    train_y = []
    for y_sample in train_data_y:
        all_labels = np.zeros(len(labels2idx))
        for labels in y_sample:
            for level, label in labels.items():
                all_labels[labels2idx[label]] = 1
        train_y.append(all_labels)

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

    return vectors_padded, np.array(train_y), token2idx


def main():
    # load train data
    train_data_x, train_data_y, labels = load_data('blurbs_train.txt')

    # create matrix
    weight_matrix, labels2idx = create_weight_matrix(n_samples=14548)

    # fill-in weight matrix
    weight_matrix = init_weight_matrix(weight_matrix, train_data_y, labels2idx)

    idx2labels = {v: k for k, v in labels2idx.items()}

    # for row, y_sample in zip(weight_matrix, train_data_y):
    #     non_zeros = np.nonzero(row)
    #     for x in non_zeros[0]:
    #         print(idx2labels[x], x)
    #     print(y_sample)
    #     print("----------")

    x_train, y_train, token2idx = build_vectors(train_data_x, train_data_y, labels2idx)

    model = build_neural_network(weight_matrix, max_input=x_train.shape[1], vocab_size=len(token2idx))

    model.summary()
    print(x_train.shape)
    print(y_train.shape)

    model.fit(x=x_train, y=y_train, validation_split=0.2, verbose=1)

if __name__ == '__main__':
    main()
