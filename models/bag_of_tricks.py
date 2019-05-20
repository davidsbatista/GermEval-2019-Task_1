import logging
from collections import Counter

import numpy as np
from gensim.models import KeyedVectors
from keras.layers import Dense, Embedding, GlobalAveragePooling1D
from keras.models import Sequential
from keras.preprocessing import sequence
from nltk import sent_tokenize, word_tokenize


class BagOfTricks:
    """
    doc-level classifier based on:

    - "Bag of tricks for efficient text classification" (A Joulin et al. 2017) EACL'17"
      see: https://www.aclweb.org/anthology/E17-2068

    """

    def __init__(self, stopwords=None):
        self.stop_words = stopwords
        self.n_top_tokens = 15000
        self.ngram_range = 3
        self.max_len = 300
        self.batch_size = 32
        self.embedding_dims = 50
        self.epochs = 10
        self.model = None
        self.max_features = None
        self.token_freq = None
        self.token2idx = None
        self.token_indice = None

    def load_static_embeddings(self, ret):
        print("Loading pre-trained static embeddings")
        static_embeddings = KeyedVectors.load(
            '/home/ubuntu/.flair/embeddings/de-wiki-fasttext-300d-1M')
        self.embedding_dims = static_embeddings.vector_size
        embedding_matrix = np.random.random((ret.max_features, self.embedding_dims))
        not_found = 0

        for idx, token in enumerate(self.token2idx):
            try:
                embedding_matrix[idx] = static_embeddings[token]
            except:
                not_found += 1

        print("not found       : ", not_found)
        print("embedding_matrix: ", embedding_matrix.shape)

        return embedding_matrix

    def build_neural_network(self, n_classes):

        model = Sequential()

        print("\nbuilding neural network")
        print("self.max_features:   ", self.max_features)
        print("self.embedding_dims: ", self.embedding_dims)

        model.add(Embedding(self.max_features, self.embedding_dims, input_length=self.max_len))
        # we add a GlobalAveragePooling1D, which will average the embeddings
        # of all words in the document
        model.add(GlobalAveragePooling1D())
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    def map_data(self, x_data, y_data=None):

        x = []
        for x_sample in x_data:
            tokens = []
            text = x_sample['title'] + " SEP " + x_sample['body']
            sentences = sent_tokenize(text, language='german')
            for s in sentences:
                tokens += word_tokenize(s)
            x.append([self.token2idx.get(token, 0) for token in tokens])

        if self.ngram_range > 1:
            # Create set of unique n-gram from the training data
            if y_data is not None:
                ngram_set = set()
                for input_list in x:
                    for i in range(2, self.ngram_range + 1):
                        set_of_ngram = self.create_ngram_set(input_list, ngram_value=i)
                        ngram_set.update(set_of_ngram)

                # Dictionary mapping n-gram token to a unique integer.
                # Integer values are greater than max_features in order
                # to avoid collision with existing features.
                start_index = self.n_top_tokens + 1
                self.token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
                indice_token = {self.token_indice[k]: k for k in self.token_indice}
                # max_features is the highest integer that could be found in the dataset.
                self.max_features = np.max(list(indice_token.keys())) + 1
                # Augmenting x with n-grams features

            x = self.add_ngram(x, self.token_indice, self.ngram_range)

        print("map data: ", self.max_features)

        x = sequence.pad_sequences(x, maxlen=self.max_len)

        return x

    def update_token_mapping(self, data_x):
        """
        Creates a token to index mapping, considering token replacements - see text_or_ent();
        The index considers only the top-'self.max_features' occurring tokens

        Parameters
        ----------
        message
        message_ent

        Returns
        -------

        """
        self.token_freq = Counter()
        self.token2idx = {}
        token_idx = 1
        for m in data_x:
            for t in m.attach_token_properties(ent_slots=y):
                token = self.text_or_ent(t)
                if token is None:
                    break
                self.token_freq[token] += 1
                if token not in self.token2idx:
                    self.token2idx[token] = token_idx
                    token_idx += 1

        self.token2idx = {k: i for i, (k, v) in
                          enumerate(self.token_freq.most_common(n=self.n_top_tokens))}

    @classmethod
    def create_ngram_set(cls, input_list, ngram_value=2):
        """
        Extract a set of n-grams from a list of integers.

        >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
        {(4, 1), (9, 4), (4, 9), (1, 4)}

        >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
        {(4, 1, 4), (4, 9, 4), (1, 4, 9), (9, 4, 1)}
        """
        return set(zip(*[input_list[i:] for i in range(ngram_value)]))

    @classmethod
    def add_ngram(cls, sequences, token_indice, ngram_range=2):
        """
        Augment the input list of list (sequences) by appending n-grams values.

        Example: adding bi-gram
        >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
        >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
        >>> add_ngram(sequences, token_indice, ngram_range=2)
        [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

        Example: adding tri-gram
        >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
        >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
        >>> add_ngram(sequences, token_indice, ngram_range=3)
        [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
        """
        new_sequences = []
        for input_list in sequences:
            new_list = input_list[:]
            for i in range(len(new_list) - ngram_range + 1):
                for ngram_value in range(2, ngram_range + 1):
                    ngram = tuple(new_list[i:i + ngram_value])
                    if ngram in token_indice:
                        new_list.append(token_indice[ngram])
            new_sequences.append(new_list)

        return new_sequences
