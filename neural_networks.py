import numpy as np

from nltk import sent_tokenize, word_tokenize

from keras import Input, Model, optimizers
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense

PADDED = 1
UNKNOWN = 0
token2idx = {}
max_sent_length = 0


def build_token_index(x_data):
    """

    :param x_data:
    :return:
    """

    # index of tokens
    global token2idx
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

    return token2idx, max_sent_length


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


def build_lstm_based_model(embeddings, label_encoder, max_sent_length):
    """
    Buils an bi-LSTM for doc. classification

    :param max_sent_length:
    :param embeddings: pre-trained static embeddings
    :param label_encoder: the enconding of the y labels
    :return:
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
