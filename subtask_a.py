#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from keras_preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize, word_tokenize
from utils.models_utils import train_bi_lstm, train_han, train_cnn_sent_class
from utils.models_utils import train_bag_of_tricks, train_logit_tf_idf
from utils.pre_processing import generate_submission_file, load_data
from utils.pre_processing import vectorize_dev_data, vectorizer


def subtask_a(train_data_x, train_data_y, dev_data_x, clf='logit'):

    data_y_level_0 = []
    for y_labels in train_data_y:
        labels_0 = set()
        for label in y_labels:
            labels_0.add(label[0])
        data_y_level_0.append(list(labels_0))
    train_data_y = data_y_level_0

    if clf == 'logit':
        model, ml_binarizer = train_logit_tf_idf(train_data_x, train_data_y, 'top_level')
        new_data_x = [x['title'] + " SEP " + x['body'] for x in dev_data_x]
        predictions = model.predict(new_data_x)
        with open('answer.txt', 'wt') as f_out:
            f_out.write(str('subtask_a\n'))
            for pred, data in zip(ml_binarizer.inverse_transform(predictions), dev_data_x):
                f_out.write(data['isbn'] + '\t' + '\t'.join([p for p in pred]) + '\n')

    else:
        if clf == 'han':
            model, ml_binarizer, max_sent_len, token2idx, max_sent, max_tokens = \
                train_han(train_data_x, train_data_y)

            processed_x = np.zeros((len(train_data_x), max_sent, max_tokens), dtype='int32')

            for i, x in enumerate(dev_data_x):
                vectorized_sentences = []
                text = x['title'] + " . " + x['body']
                sentences = sent_tokenize(text, language='german')
                for s in sentences:
                    vectorized_sentences.append(
                        vectorizer(word_tokenize(s, language='german'), token2idx))

                padded_sentences = pad_sequences(vectorized_sentences, padding='post',
                                                 truncating='post', maxlen=max_tokens,
                                                 value=token2idx['PADDED'])

                pad_size = max_sent - padded_sentences.shape[0]

                if pad_size < 0:
                    padded_sentences = padded_sentences[0:max_sent]
                else:
                    padded_sentences = np.pad(padded_sentences, ((0, pad_size), (0, 0)),
                                              mode='constant',
                                              constant_values=0)

                # Store this observation as the i-th observation in the data matrix
                processed_x[i] = padded_sentences[None, ...]

            print(processed_x.shape)

            # test_vectors = vectorize_dev_data(dev_data_x, max_sent_len, token2idx)
            predictions = model.predict(processed_x)

        if clf == 'lstm':
            model, ml_binarizer, max_sent_len, token2idx = train_bi_lstm(train_data_x, train_data_y)
            test_vectors = vectorize_dev_data(dev_data_x, max_sent_len, token2idx)
            predictions = model.predict(test_vectors)

        if clf == 'cnn':
            model, ml_binarizer, max_sent_len, token2idx = train_cnn_sent_class(train_data_x,
                                                                                train_data_y,
                                                                                'top_level')
            test_vectors = vectorize_dev_data(dev_data_x, max_sent_len, token2idx)
            predictions = model.predict(test_vectors)

        if clf == 'bag-of-tricks':
            model, ml_binarizer, max_sent_len, token2idx = train_bag_of_tricks(train_data_x,
                                                                               train_data_y)

            test_vectors = vectorize_dev_data(dev_data_x, max_sent_len, token2idx)
            predictions = model.predict(test_vectors)

        binary_predictions = []
        for pred, true in zip(predictions, dev_data_x):
            binary = [0 if i <= 0.5 else 1 for i in pred]
            binary_predictions.append(binary)

        generate_submission_file(np.array(binary_predictions), ml_binarizer, dev_data_x)


def main():
    # load train data
    train_data_x, train_data_y, labels = load_data('blurbs_train.txt', dev=True)

    # load dev data
    dev_data_x, _, _ = load_data('blurbs_dev_participants.txt', dev=True)

    # train subtask_a
    # subtask_a(train_data_x, train_data_y, dev_data_x, clf='bag-of-tricks')
    # subtask_a(train_data_x, train_data_y, dev_data_x, clf='han')
    # subtask_a(train_data_x, train_data_y, dev_data_x, clf='lstm')
    # subtask_a(train_data_x, train_data_y, dev_data_x, clf='logit')
    subtask_a(train_data_x, train_data_y, dev_data_x, clf='cnn')


if __name__ == '__main__':
    main()