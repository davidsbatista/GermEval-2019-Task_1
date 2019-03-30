from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence, \
    DocumentRNNEmbeddings
from nltk import sent_tokenize, word_tokenize


def convert_data_flair_format(train_x):
    pass


def embed_document(sample_x):
    # initialize the word embeddings
    glove_embedding = WordEmbeddings('glove')
    flair_embedding_forward = FlairEmbeddings('german-forward')
    flair_embedding_backward = FlairEmbeddings('german-backward')

    # initialize the document embeddings, mode = mean
    document_pool_embeddings = DocumentPoolEmbeddings([glove_embedding,
                                                       flair_embedding_backward,
                                                       flair_embedding_forward])

    document_rnn_embeddings = DocumentRNNEmbeddings(
        [glove_embedding, flair_embedding_backward, flair_embedding_forward],
        rnn_type='LSTM',
        bidirectional=True)

    # create an example sentence
    text = 'Ein Blick hinter die Kulissen eines Krankenhauses vom Autor der Bestseller ' \
           '"Der Medicus" und "Der Medicus von Saragossa". Der Wissenschaftler Adam Silverstone, ' \
           'der kubanische Aristokrat Rafael Meomartino und der Farbige Spurgeon Robinson - ' \
           'sie sind drei grundverschiedene Klinik-Ärzte, die unter der unerbittlichen Aufsicht ' \
           'von Dr. Longwood praktizieren. Eines Tages stirbt eine Patientin, und Dr. Longwood ' \
           'wittert einen Behandlungsfehler. Sofort macht er sich auf die Suche nach ' \
           'einem Schuldigen, dem er die Verantwortung in die Schuhe schieben könnte'

    tokens = []
    sentences = sent_tokenize(text, language='german')
    for s in sentences:
        tokens = word_tokenize(s.lower())
        document_rnn_embeddings.embed(' '.join(tokens))

    # embed the sentence with our document embedding
    document_pool_embeddings.embed(sentence)

    # now check out the embedded sentence.
    print(sentence.get_embedding())

    # embed the sentence with our document embedding
    document_rnn_embeddings.embed(sentence)

    # now check out the embedded sentence.
    print(sentence.get_embedding())
