from flair.data import TaggedCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence, \
    DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from nltk import sent_tokenize, word_tokenize
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer


def embed_documents(train_x, test_x, train_y, test_y, dev_data_x):

    train_data_x = []
    for x, y in zip(train_x, train_y):
        # a single embedding for the whole document
        tokens = word_tokenize(x['title'].lower() + " SEP " + x['body'].lower())
        if len(tokens) == 0:
            continue
        flair_sentence = Sentence(' '.join(tokens))
        flair_sentence.add_labels(y)
        train_data_x.append(flair_sentence)

    test_data_x = []
    test_data_y = []
    for x, y in zip(test_x, test_y):
        # a single embedding for the whole document
        tokens = word_tokenize(x['title'].lower() + " SEP " + x['body'].lower())
        if len(tokens) == 0:
            continue
        flair_sentence = Sentence(' '.join(tokens))
        flair_sentence.add_labels(y)
        test_data_x.append(flair_sentence)
        test_data_y.append(y)

    corpus = TaggedCorpus(train=train_data_x, dev=test_data_x, test=[])
    # stats = corpus.obtain_statistics()
    # print(stats)
    label_dict = corpus.make_label_dictionary()

    word_embeddings = [WordEmbeddings('de-crawl')
                       FlairEmbeddings('de-forward'),
                       FlairEmbeddings('de-backward')]

    document_embeddings = DocumentRNNEmbeddings(word_embeddings,
                                                hidden_size=64,
                                                reproject_words=False,
                                                reproject_words_dimension=256,
                                                dropout=0.5,
                                                word_dropout=0.3,
                                                locked_dropout=0.25,
                                                bidirectional=True,
                                                rnn_type='LSTM')

    classifier = TextClassifier(document_embeddings,
                                label_dictionary=label_dict,
                                multi_label=True)

    trainer = ModelTrainer(classifier, corpus)
    trainer.train('resources/taggers/',
                  learning_rate=0.01,
                  mini_batch_size=32,
                  anneal_factor=0.5,
                  patience=5,
                  max_epochs=5)

    # 8. plot training curves (optional)
    # from flair.visual.training_curves import Plotter
    # plotter = Plotter()
    # plotter.plot_training_curves('resources/taggers/germeval19/loss.tsv')
    # plotter.plot_weights('resources/taggers/germeval19/weights.txt')
    classifier.save("text-classifier-model")

    # print(corpus.obtain_statistics())
    predictions = classifier.predict(test_data_x)
    pred_labels = []
    for sent in predictions:
        preds = []
        for x in sent.labels:
            preds.append(x.value)
        pred_labels.append(preds)

    ml_binarizer = MultiLabelBinarizer()
    true_y_labels = ml_binarizer.fit_transform(test_data_y)
    pred_y_labels = ml_binarizer.transform(pred_labels)

    report = classification_report(true_y_labels, pred_y_labels, target_names=ml_binarizer.classes_)

    print(report)

    # apply trained classifier on dev data
    dev_data = []
    for x in dev_data_x:
        # a single embedding for the whole document
        tokens = word_tokenize(x['title'].lower() + " SEP " + x['body'].lower())
        if len(tokens) == 0:
            continue
        flair_sentence = Sentence(' '.join(tokens))
        dev_data.append(flair_sentence)

    predictions = classifier.predict(dev_data)
    pred_labels = []
    for sent in predictions:
        preds = []
        for x in sent.labels:
            preds.append(x.value)
        pred_labels.append(preds)

    with open('answer_a.txt', 'wt') as f_out:
        f_out.write(str('subtask_a\n'))
        for pred, data in zip(pred_labels, dev_data_x):
            f_out.write(data['isbn'] + '\t' + '\t'.join([p for p in pred]) + '\n')

    """
    for x in data_x:
        # # one embedding per sentence
        # sentences = sent_tokenize(x, language='german')
        # flair_sentences = []
        # for s in sentences:
        #     tokens = word_tokenize(s.lower())
        #     flair_sentences.append(Sentence(' '.join(tokens)))
        # document_rnn_embeddings.embed(flair_sentences)
    """
