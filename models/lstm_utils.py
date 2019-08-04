import numpy as np
from keras import Input, Model, optimizers, regularizers, constraints
from keras.engine import Layer
from keras.layers import Bidirectional, CuDNNLSTM, Dense, Dropout, Embedding, LSTM, initializers, K


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))),
                        (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


def biLSTM(maxlen, max_features, embed_size, embedding_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(LSTM(300, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(x)
    # x = Bidirectional(CuDNNLSTM(300, return_sequences=True))(x)
    x = Attention(maxlen)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(6, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)

    return model


def make_glovevec(glovepath, max_features, embed_size, word_index, veclen=300):
    embeddings_index = {}
    f = open(glovepath)
    for line in f:
        values = line.split()
        word = ' '.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        embeddings_index[word] = coefs.reshape(-1)
    f.close()

    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def build_lstm_based_model(embeddings, label_encoder, max_sent_length, token2idx):
    """
    Buils an bi-LSTM for doc. classification

    :param max_sent_length:
    :param embeddings: pre-trained static embeddings
    :param label_encoder: the encoding of the y labels
    :return:
    """
    hidden_units = 128
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
    l_lstm = Bidirectional(LSTM(hidden_units))(embedded_sequences)
    l_lstm_w_drop = Dropout(dense_dropout)(l_lstm)
    preds = Dense(len(label_encoder.classes_),
                  activation='sigmoid', name='sigmoid')(l_lstm_w_drop)
    model = Model(inputs=[sequence_input], outputs=[preds])

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(lr=learning_rate),
                  metrics=['acc'])

    print('{} out of {} words randomly initialized'.format(not_found, len(token2idx)))

    return model
