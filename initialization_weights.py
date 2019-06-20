#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import numpy as np
import tensorflow as tf
from keras import Input, Model, backend as K
from keras.engine.saving import load_model
from keras.layers import AlphaDropout, Concatenate, Convolution1D, Dense, Embedding, \
    GlobalMaxPooling1D
from keras_preprocessing.sequence import pad_sequences
from nltk import sent_tokenize, word_tokenize

from data_analysis import extract_hierarchy
from models.utils import build_token_index, vectorize_dev_data, vectorizer
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
    embedding_size = 1
    conv_layers = [[256, 10], [256, 7], [256, 5], [256, 3]]
    fully_connected_layers = [weight_matrix.shape[0], weight_matrix.shape[0]]
    dropout_p = 0.1
    num_of_classes = weight_matrix.shape[1]
    optimizer = "adam"
    loss = "binary_crossentropy"
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
        # x = Dense(fl, activation='selu', kernel_initializer='lecun_normal')(x)
        x = Dense(fl, activation='tanh', kernel_initializer='random_uniform')(x)
        x = AlphaDropout(dropout_p)(x)

    # Output layer
    predictions = Dense(num_of_classes, activation='sigmoid')(x)

    # Build and compile model
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    from keras.utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    return model


def build_vectors(train_data_x, train_data_y, labels2idx):

    # ToDo: vectorize input data and target

    token2idx, max_sent_len, _ = build_token_index(train_data_x)

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

    return vectors_padded, np.array(train_y), token2idx, max_sent_len


def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)


def init_f(shape, dtype=None):
    ker = np.zeros(shape, dtype=dtype)
    ker[tuple(map(lambda x: int(np.floor(x/2)), ker.shape))]=1
    return tf.convert_to_tensor(ker)


def main():

    # load train data
    train_data_x, train_data_y, labels = load_data('blurbs_train.txt', dev=True)

    # create matrix
    weight_matrix, labels2idx = create_weight_matrix(n_samples=len(train_data_x))

    # fill-in weight matrix
    weight_matrix = init_weight_matrix(weight_matrix, train_data_y, labels2idx)
    x_train, y_train, token2idx, max_sent_len = build_vectors(train_data_x, train_data_y, labels2idx)

    if not os.path.exists('global_classifier.h5'):
        model = build_neural_network(weight_matrix, max_input=x_train.shape[1], vocab_size=len(token2idx))
        model.summary()
        model.fit(x=x_train, y=y_train, validation_split=0.2, verbose=1, epochs=5)
        model.save('global_classifier.h5')  # creates a HDF5 file 'my_model.h5'

    else:
        model = load_model(filepath='global_classifier.h5')

        # load dev data
        dev_data_x, _, _ = load_data('blurbs_dev_participants.txt', dev=True)

    dev_vector = vectorize_dev_data(dev_data_x, max_sent_len, token2idx)
    predictions = model.predict(dev_vector, verbose=1)

    filtered = np.array(len(labels2idx) * [0.3])
    pred_bin = (predictions > filtered).astype(int)

    idx2labels = {v: k for k, v in labels2idx.items()}

    # ToDo: save to file to allow to perform evaluation

    for row_pred, sample in zip(pred_bin, dev_data_x):
        print(sample['isbn'], end='\t')
        if np.count_nonzero(row_pred) > 0:
            for x in np.nditer(np.nonzero(row_pred)):
                print(idx2labels[int(x)], end='\t')
        print()


if __name__ == '__main__':
    main()
