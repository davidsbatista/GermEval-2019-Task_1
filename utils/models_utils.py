from collections import defaultdict

import nltk
import numpy as np
from gensim.models import KeyedVectors
from keras import Sequential
from keras.legacy import layers
from keras_preprocessing.sequence import pad_sequences
from nltk import sent_tokenize, word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from models.bag_of_tricks import BagOfTricks
from models.convnets_utils import get_cnn_pre_trained_embeddings, get_embeddings_layer
from models.keras_han.model import HAN
from models.lstm_utils import build_lstm_based_model
from utils.pre_processing import build_token_index, tokenise, vectorizer


def train_bi_lstm(train_data_x, train_data_y, tokenisation):
    """
    Trains a biLSTM classifier, message is represented by the concatenation of the two last
    states from each LSTM.

    :param train_data_x:
    :param train_data_y:
    :return:
    """
    low = tokenisation['low']
    simple = tokenisation['simple']
    stop = tokenisation['stop']

    token2idx, max_sent_len, _ = build_token_index(train_data_x,
                                                   lowercase=low,
                                                   simple=simple,
                                                   remove_stopwords=stop)

    # vectorize, i.e. tokens to indexes and pad
    print("Vectorizing input data\n")
    vectors = []
    for x in train_data_x:
        text = x['title'] + " SEP " + x['body']
        tokens = tokenise(text, lowercase=low, simple=simple, remove_stopwords=stop)
        vector = vectorizer(tokens, token2idx)
        vectors.append(vector)
    vectors_padded = pad_sequences(vectors,
                                   padding='post',
                                   maxlen=max_sent_len,
                                   truncating='post',
                                   value=token2idx['PADDED'])

    # y_data: encode into one-hot vectors
    ml_binarizer = MultiLabelBinarizer()
    y_labels = ml_binarizer.fit_transform(train_data_y)
    print('Total of {} classes'.format(len(ml_binarizer.classes_)))

    train_data_x = vectors_padded
    data_y = y_labels

    # split into train and hold out set
    # train_x, test_x, train_y, test_y = train_test_split(train_data_x, data_y,
    #                                                    random_state=42,
    #                                                    test_size=0.30)

    print("Loading pre-trained Embeddings\n")
    static_embeddings = KeyedVectors.load('resources/de-wiki-fasttext-300d-1M')
    model = build_lstm_based_model(static_embeddings, ml_binarizer, max_sent_len, token2idx)

    # since we have imbalanced dataset
    # sample_weights = compute_sample_weight('balanced', train_y)
    model.fit(train_data_x, data_y, batch_size=16, epochs=10, verbose=1, validation_split=0.2)

    # predictions = model.predict(test_x)

    # ToDo: there must be a more efficient way to do this
    # binary_predictions = []
    # for pred in predictions:
    #    binary_predictions.append([0 if i <= 0.5 else 1 for i in pred])
    # print(classification_report(test_y, np.array(binary_predictions),
    #                            target_names=ml_binarizer.classes_))

    return model, ml_binarizer, max_sent_len, token2idx


def train_han(train_data_x, train_data_y, tokenisation):

    low = tokenisation['low']
    simple = tokenisation['simple']
    stop = tokenisation['stop']

    token2idx, max_sent_len, _ = build_token_index(train_data_x,
                                                   lowercase=low,
                                                   simple=simple,
                                                   remove_stopwords=stop)

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

    han_model.fit(train_x, train_y, batch_size=16, epochs=20, validation_split=0.2)

    predictions = han_model.predict(test_x, verbose=1)

    binary_predictions = []
    for pred in predictions:
        binary_predictions.append([0 if i <= 0.5 else 1 for i in pred])
    print(classification_report(test_y, np.array(binary_predictions),
                                target_names=ml_binarizer.classes_))

    return han_model, ml_binarizer, max_sent_len, token2idx, max_sent, max_tokens


def train_bag_of_tricks(train_data_x, train_data_y, tokenisation):
    bot = BagOfTricks()
    n_top_tokens = 100000

    low = tokenisation['low']
    simple = tokenisation['simple']
    stop = tokenisation['stop']

    token2idx, max_sent_len, token_freq = build_token_index(train_data_x,
                                                            lowercase=low,
                                                            simple=simple,
                                                            remove_stopwords=stop)

    # select only top-k tokens
    print("total nr. of tokens  : ", len(token2idx))
    token2idx = {k: i for i, (k, v) in enumerate(token_freq.most_common(n=n_top_tokens))}
    print("selected top-k tokens: ", len(token2idx))
    print("max_sent_length      : ", max_sent_len)

    PADDED = 1
    UNKNOWN = 0
    token2idx["PADDED"] = PADDED
    token2idx["UNKNOWN"] = UNKNOWN

    bot.token2idx = token2idx
    bot.max_len = max_sent_len

    # map data to vectors of n-grams
    train_data_x = bot.map_data(train_data_x, train_data_y)

    # y_data: encode into one-hot vectors
    ml_binarizer = MultiLabelBinarizer()
    data_y = ml_binarizer.fit_transform(train_data_y)
    print('Total of {} classes'.format(len(ml_binarizer.classes_)))
    n_classes = len(ml_binarizer.classes_)

    """
    # split into train and hold out set
    train_x, test_x, train_y, test_y = train_test_split(train_data_x, data_y, random_state=42,
                                                        test_size=0.30)
    print(train_x.shape)
    print(train_y.shape)

    # build a neural network and train a model
    model = bot.build_neural_network(n_classes)
    model.fit(train_x, train_y, batch_size=32, epochs=30, verbose=1)

    predictions = model.predict([test_x], verbose=1)

    binary_predictions = []
    for pred in predictions:
        binary_predictions.append([0 if i <= 0.5 else 1 for i in pred])
    report = classification_report(test_y, np.array(binary_predictions),
                                   target_names=ml_binarizer.classes_)
    print(report)

    with open('classification_report.txt', 'at+') as f_out:
        f_out.write(report)
        f_out.write('\n')
    """

    # train on all data
    model = bot.build_neural_network(n_classes)

    print("model.stop_words  :  ", bot.stop_words)
    print("model.n_top_tokens:  ", bot.n_top_tokens)
    print("model.ngram_range :  ", bot.ngram_range)
    print("model.max_len     :  ", bot.max_len)
    print("model.batch_size  :  ", bot.batch_size)
    print("embedding_dims    :  ", bot.embedding_dims)
    print("model.max_features:  ", bot.max_features)
    print("model.token_freq  :  ", bot.token_freq)
    print("model.token2idx   :  ", len(bot.token2idx))
    print("model.token_indice:  ", len(bot.token_indice))

    model.fit(train_data_x, data_y, batch_size=16, epochs=10, verbose=1, validation_split=0.2)

    return model, ml_binarizer, max_sent_len, token2idx


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

    # NOTE: simple tokenisation using TfidfVectorizer regex give better results
    #       than NLTK german specific
    # de_stemmer = GermanStemmer()
    # all_doc_tokens = []
    # for x in new_data_x:
    #     doc_tokens = []
    #     for s in sent_tokenize(x, language='german'):
    #         tokens = wordpunct_tokenize(s)
    #         words = [w.lower() for w in nltk.Text(tokens) if w.isalpha()]
    #         doc_tokens.extend(words)
    #     doc_tokens_stemmed = [de_stemmer.stem(x) for x in doc_tokens]
    #     all_doc_tokens.append(doc_tokens_stemmed)
    #     all_doc_tokens.append(doc_tokens)
    # new_data_x = all_doc_tokens

    # split into train and hold out set
    train_x, test_x, train_y, test_y = train_test_split(new_data_x, data_y, random_state=42,
                                                        test_size=0.30)

    stop_words = set(stopwords.words('german'))
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=None,
                                  ngram_range=(2, 5),
                                  max_df=0.75,
                                  analyzer='char')),
        ('clf', OneVsRestClassifier(
            LogisticRegression(class_weight='balanced', solver='sag', max_iter=5000),
            n_jobs=10))
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

    # train a classifier on all data using the parameters that yielded best result
    print("Training classifier with best parameters on all data")
    best_tf_idf = grid_search_tune.best_estimator_.steps[0][1]
    clf = grid_search_tune.best_estimator_.steps[1][1]

    best_pipeline = Pipeline([('tfidf', best_tf_idf), ('clf', clf)])
    best_pipeline.fit(new_data_x, data_y)

    return best_pipeline, ml_binarizer


def train_naive_bayes(train_data_x, train_data_y, level_label):
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

    # simple tokenization using TfidfVectorizer regex works better than NLTK german specific
    new_data_x = [x['title'] + " SEP " + x['body'] for x in train_data_x]

    # split into train and hold out set
    train_x, test_x, train_y, test_y = train_test_split(new_data_x, data_y, random_state=42,
                                                        test_size=0.30)

    stop_words = set(stopwords.words('german'))
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 2), max_df=0.75)),
        ('clf', OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None))),
    ])
    parameters = {
        'tfidf__max_df': (0.25, 0.5, 0.75),
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'clf__estimator__alpha': (1e-2, 1e-3)
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
        f_out.write(level_label + '\n')
        f_out.write("=" * len(level_label) + '\n')
        f_out.write(report)
        f_out.write('\n')

    # train a classifier on all data using the parameters that yielded best result
    print("Training classifier with best parameters on all data")
    best_tf_idf = grid_search_tune.best_estimator_.steps[0][1]
    clf = grid_search_tune.best_estimator_.steps[1][1]

    best_pipeline = Pipeline([('tfidf', best_tf_idf), ('clf', clf)])
    best_pipeline.fit(new_data_x, data_y)

    return best_pipeline, ml_binarizer


def train_cnn_sent_class(train_data_x, train_data_y, tokenisation):

    low = tokenisation['low']
    simple = tokenisation['simple']
    stop = tokenisation['stop']

    token2idx, max_sent_len, _ = build_token_index(train_data_x,
                                                   lowercase=low,
                                                   simple=simple,
                                                   remove_stopwords=stop)

    # vectorize, i.e. tokens to indexes and pad
    print("Vectorizing input data\n")
    vectors = []
    for x in train_data_x:
        text = x['title'] + " SEP " + x['body']
        tokens = tokenise(text, lowercase=low, simple=simple, remove_stopwords=stop)
        vector = vectorizer(tokens, token2idx)
        vectors.append(vector)
    vectors_padded = pad_sequences(vectors,
                                   padding='post',
                                   maxlen=max_sent_len,
                                   truncating='post',
                                   value=token2idx['PADDED'])

    # encode target into one-hot vectors
    ml_binarizer = MultiLabelBinarizer()
    y_labels = ml_binarizer.fit_transform(train_data_y)
    print('Total of {} classes'.format(len(ml_binarizer.classes_)))
    n_classes = len(ml_binarizer.classes_)
    train_data_x = vectors_padded
    data_y = y_labels

    """
    # split into train and hold out set
    train_x, test_x, train_y, test_y = train_test_split(train_data_x, data_y,
                                                        random_state=42,
                                                        test_size=0.30)
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

    """
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
    """

    # train on all data without validation split
    embedding_layer = get_embeddings_layer(embedding_matrix, 'static-embeddings',
                                           max_sent_len, trainable=True)
    model = get_cnn_pre_trained_embeddings(embedding_layer, max_sent_len, n_classes)
    model.fit(train_data_x, data_y, batch_size=16, epochs=5, verbose=True)

    return model, ml_binarizer, max_sent_len, token2idx


def train_cnn_sent_class_grid_search(train_data_x, train_data_y, tokenisation):
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

    """        
    low = tokenisation['low']
    simple = tokenisation['simple']
    stop = tokenisation['stop']

    token2idx, max_sent_len, _ = build_token_index(train_data_x,
                                                   lowercase=low,
                                                   simple=simple,
                                                   remove_stopwords=stop)

    # vectorize, i.e. tokens to indexes and pad
    print("Vectorizing input data\n")
    vectors = []
    for x in train_data_x:
        text = x['title'] + " SEP " + x['body']
        tokens = tokenise(text, lowercase=low, simple=simple, remove_stopwords=stop)
        vector = vectorizer(tokens, token2idx)
        vectors.append(vector)
    vectors_padded = pad_sequences(vectors,
                                   padding='post',
                                   maxlen=max_sent_len,
                                   truncating='post',
                                   value=token2idx['PADDED'])

    # encode target into one-hot vectors
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

    def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
        model = Sequential()
        model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
        model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import RandomizedSearchCV

    # Main settings
    epochs = 20
    embedding_dim = 50
    maxlen = 100
    output_file = 'data/output.txt'

    # sentences = df['sentence'].values
    # y = df['label'].values


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