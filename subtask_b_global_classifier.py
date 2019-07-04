#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import tensorflow as tf
import random as rn

# necessary for starting Numpy generated random numbers in a well-defined initial state.
from gensim.models import KeyedVectors

from models.convnets_utils import get_embeddings_layer

np.random.seed(42)

# necessary for starting core Python generated random numbers in a well-defined state.
rn.seed(12345)

# force TensorFlow to use single thread, multiple-threads can lead to non-reproducible results.
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K
# make random number generation in the TensorFlow backend have a well-defined initial state.
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=None)
K.set_session(sess)

import os
import tensorflow as tf
from keras import Input, Model, backend as K
from keras.engine.saving import load_model
from keras.layers import AlphaDropout, Concatenate, Convolution1D, Dense, Embedding, \
    GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, concatenate
from keras_preprocessing.sequence import pad_sequences


from utils.pre_processing import build_token_index, vectorizer, load_data, vectorize_dev_data, \
    tokenise
from utils.statistical_analysis import extract_hierarchy


def write_submission_file(dev_data_x, idx2labels, pred_bin):
    with open('answer.txt', 'wt') as f_out:
        # subtask-a
        f_out.write(str('subtask_a\n'))
        for row_pred, sample in zip(pred_bin, dev_data_x):
            f_out.write(sample['isbn'] + '\t')
            if np.count_nonzero(row_pred) > 0:
                for x in np.nditer(np.nonzero(row_pred)):
                    if int(x) <= 7:
                        label = idx2labels[int(x)]
                        f_out.write(label + '\t')
            f_out.write('\n')

        # subtask-b
        f_out.write(str('subtask_b\n'))
        for row_pred, sample in zip(pred_bin, dev_data_x):
            f_out.write(sample['isbn'] + '\t')
            if np.count_nonzero(row_pred) > 0:
                for x in np.nditer(np.nonzero(row_pred)):
                    label = idx2labels[int(x)]
                    f_out.write(label + '\t')
            f_out.write('\n')


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
    embedding_size = 100
    # conv_layers = [[256, 10], [256, 7], [256, 5], [256, 3]]
    conv_layers = [[300, 1], [300, 2], [300, 3]]
    fully_connected_layers = [weight_matrix.shape[0], weight_matrix.shape[0]]
    # dropout_p = 0.1
    dropout_p = 0.5
    num_of_classes = weight_matrix.shape[1]
    optimizer = "adam"
    loss = "binary_crossentropy"
    threshold = 1e-6

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
        x = Dense(fl, activation='relu', kernel_initializer='random_uniform')(x)
        x = AlphaDropout(dropout_p)(x)

    # Output layer
    predictions = Dense(num_of_classes, activation='sigmoid')(x)

    # Build and compile model
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


def build_neural_network_2(weight_matrix, max_sent_len, vocab_size, token2idx):

    num_of_classes = weight_matrix.shape[1]

    # embeddings layer
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

    # connect the input with the embedding layer
    i = Input(shape=(max_sent_len,), dtype='int32', name='main_input')
    x_input = embedding_layer(i)

    # generate several branches in the network, each for a different convolution+pooling operation,
    # and concatenate the result of each branch into a single vector
    n_grams = [3, 5, 7, 10]
    feature_maps = 256
    branches = []
    suffix = 'dynamic_embeddings'
    for n in n_grams:
        branch = Conv1D(filters=feature_maps, kernel_size=n, activation='relu',
                        # kernel_regularizer=regularizers.l2(0.01),
                        name='Conv_' + suffix + '_' + str(n))(x_input)

        branch = MaxPooling1D(pool_size=(max_sent_len - n + 1),
                              strides=None, padding='valid',
                              name='MaxPooling_' + suffix + '_' + str(n))(branch)
        branch = Flatten(name='Flatten_' + suffix + '_' + str(n))(branch)
        branches.append(branch)

    z = concatenate(branches, axis=-1)
    x = Dense(weight_matrix.shape[0], activation='selu', kernel_initializer='random_uniform')(z)

    # pass the concatenated vector to the prediction layer
    o = Dense(num_of_classes, activation='sigmoid', name='output')(x)

    model = Model(inputs=i, outputs=o)
    model.compile(loss={'output': 'binary_crossentropy'}, optimizer='adam', metrics=['accuracy'])

    """
    model.fit(train_x, train_y, batch_size=16, epochs=20, verbose=True, validation_split=0.33)
    predictions = model.predict([test_x], verbose=1)
    # train on all data without validation split
    embedding_layer = get_embeddings_layer(embedding_matrix, 'static-embeddings',
                                           max_sent_len, trainable=True)
    model = get_cnn_pre_trained_embeddings(embedding_layer, max_sent_len, n_classes)
    model.fit(train_data_x, data_y, batch_size=16, epochs=5, verbose=True)
    """

    return model


def build_vectors(train_data_x, train_data_y, labels2idx, tokenisation):

    low = tokenisation['low']
    simple = tokenisation['simple']
    stop = tokenisation['stop']

    token2idx, max_sent_len, _ = build_token_index(train_data_x,
                                                   lowercase=low,
                                                   simple=simple,
                                                   remove_stopwords=stop)

    print("token2idx: ", len(token2idx))

    # y_data: encode into one-hot vectors with all labels in the hierarchy
    train_y = []
    for y_sample in train_data_y:
        all_labels = np.zeros(len(labels2idx))
        for labels in y_sample:
            for level, label in labels.items():
                all_labels[labels2idx[label]] = 1
        train_y.append(all_labels)

    # vectorize, i.e. tokens to indexes and pad
    print("Vectorizing input data\n")
    vectors = []
    for x in train_data_x:
        text = x['title'] + " SEP " + x['body']
        tokens = tokenise(text, lowercase=low, simple=simple, remove_stopwords=stop)
        vector = vectorizer(tokens, token2idx)
        vectors.append(vector)
    vectors_padded = pad_sequences(vectors, padding='post', maxlen=max_sent_len,
                                   truncating='post', value=token2idx['PADDED'])

    return vectors_padded, np.array(train_y), token2idx, max_sent_len


def my_init(shape, dtype=None):
    # ToDo
    return K.random_normal(shape, dtype=dtype)


def init_f(shape, dtype=None):
    ker = np.zeros(shape, dtype=dtype)
    ker[tuple(map(lambda x: int(np.floor(x / 2)), ker.shape))] = 1
    return tf.convert_to_tensor(ker)


def main():
    # load train data
    train_data_x, train_data_y, labels = load_data('blurbs_train.txt', dev=True)

    # load dev data
    dev_data_x, _, _ = load_data('blurbs_dev_participants.txt', dev=True)

    # create matrix and fill-in weight matrix
    weight_matrix, labels2idx = create_weight_matrix(n_samples=len(train_data_x))
    weight_matrix = init_weight_matrix(weight_matrix, train_data_y, labels2idx)

    # tokenise training data
    tokenisation = {'low': True, 'simple': True, 'stop': True}
    x_train, y_train, token2idx, max_sent_len = build_vectors(train_data_x, train_data_y,
                                                              labels2idx, tokenisation)

    if not os.path.exists('global_classifier.h5'):
        model = build_neural_network(weight_matrix,
                                     max_input=x_train.shape[1],
                                     vocab_size=len(token2idx))
        model.summary()
        model.fit(x=x_train, y=y_train,
                  batch_size=128,
                  shuffle=True,
                  validation_split=0.4,
                  verbose=1,
                  epochs=250)
        model.save('global_classifier.h5')
    else:
        model = load_model(filepath='global_classifier.h5')

    dev_vector = vectorize_dev_data(dev_data_x, max_sent_len, token2idx, tokenisation)

    predictions = model.predict([dev_vector], verbose=1)

    for x in predictions:
        print(np.nonzero(x))
    print(dev_vector.shape)

    # ToDo:
    # - tune threshold for different levels?
    #   0-7 top-level: 0.5
    #   8-X 1st_level: 0.3
    #   X-343 2nd_level: 0.1
    # - initialize weigh matrix

    filtered = np.array(len(labels2idx) * [0.001])
    pred_bin = (predictions > filtered).astype(int)
    idx2labels = {v: k for k, v in labels2idx.items()}
    write_submission_file(dev_data_x, idx2labels, pred_bin)


if __name__ == '__main__':
    main()
