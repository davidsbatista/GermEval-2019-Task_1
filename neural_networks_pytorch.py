from flair.data import TaggedCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence, \
    DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from nltk import sent_tokenize, word_tokenize


def convert_data_flair_format(train_x):
    pass


def embed_documents(train_x, test_x, train_y, test_y, dev_data_x):
    train_data_x = []
    for x, y in zip(train_x, train_y):
        # a single embedding for the whole document
        tokens = word_tokenize(x['body'].lower())
        #print(x['body'].lower())
        #print(tokens)
        #print(len(tokens))
        if len(tokens) == 0:
            continue
        try:
            flair_sentence = Sentence(' '.join(tokens))
            flair_sentence.add_labels(y)
            train_data_x.append(flair_sentence)
        except UnboundLocalError:
            print(x)

    test_data_x = []
    for x, y in zip(test_x, test_y):
        # a single embedding for the whole document
        tokens = word_tokenize(x['body'].lower())
        if len(tokens) == 0:
            continue
        flair_sentence = Sentence(' '.join(tokens))
        flair_sentence.add_labels(y)
        test_data_x.append(flair_sentence)

    corpus = TaggedCorpus(train=train_data_x, test=test_data_x, dev=test_data_x)
    stats = corpus.obtain_statistics()
    print(stats)
    label_dict = corpus.make_label_dictionary()

    word_embeddings = [WordEmbeddings('de-crawl')]
                       # FlairEmbeddings('de-forward'),
                       # FlairEmbeddings('de-backward')]

    # 4. initialize document embedding by passing list of word embeddings
    # Can choose between many RNN types (GRU by default, to change use rnn_type parameter)
    document_embeddings = DocumentRNNEmbeddings(word_embeddings,
                                                hidden_size=512,
                                                reproject_words=True,
                                                reproject_words_dimension=256,
                                                bidirectional=True
                                                )

    # 5. create the text classifier
    classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, multi_label=True)

    # 6. initialize the text classifier trainer
    trainer = ModelTrainer(classifier, corpus)

    # 7. start the training
    trainer.train('resources/taggers/',
                  learning_rate=0.1,
                  mini_batch_size=32,
                  anneal_factor=0.5,
                  patience=5,
                  max_epochs=25)

    # 8. plot training curves (optional)
    # from flair.visual.training_curves import Plotter
    # plotter = Plotter()
    # plotter.plot_training_curves('resources/taggers/germeval19/loss.tsv')
    # plotter.plot_weights('resources/taggers/germeval19/weights.txt')
    classifier.save("text-classifier-model")

    # print(corpus.obtain_statistics())

    dev_data = []
    for x in dev_data_x:
        # a single embedding for the whole document
        tokens = word_tokenize(x['body'].lower())
        flair_sentence = Sentence(' '.join(tokens))
        dev_data.append(flair_sentence)

    predictions = classifier.predict(dev_data)

    for sent in predictions:
        print(sent)
        print(sent.labels)
        print()

    """
    # initialize the word embeddings
    glove_embedding = WordEmbeddings('glove')
    flair_embedding_forward = FlairEmbeddings('german-forward')
    flair_embedding_backward = FlairEmbeddings('german-backward')

    # initialize the document embeddings, mode = mean
    document_pool_embeddings = DocumentPoolEmbeddings([glove_embedding,
                                                       flair_embedding_backward,
                                                       flair_embedding_forward])

    # initialize the document embeddings, mode = LSTM
    document_rnn_embeddings = DocumentRNNEmbeddings(
        [glove_embedding, flair_embedding_backward, flair_embedding_forward],
        rnn_type='LSTM',
        bidirectional=True)

    for x in data_x:
        # a single embedding for the whole document
        tokens = word_tokenize(x['body'].lower())
        flair_sentence = Sentence(' '.join(tokens))

        # document_rnn_embeddings.embed(flair_sentence)

        print(flair_sentence.tokens)
        print(flair_sentence.get_embedding())
        print()

        # # one embedding per sentence
        # sentences = sent_tokenize(x, language='german')
        # flair_sentences = []
        # for s in sentences:
        #     tokens = word_tokenize(s.lower())
        #     flair_sentences.append(Sentence(' '.join(tokens)))
        # document_rnn_embeddings.embed(flair_sentences)
    """
