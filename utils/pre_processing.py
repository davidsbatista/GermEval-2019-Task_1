import datetime
from collections import Counter, defaultdict
from datetime import datetime
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup, NavigableString, Tag
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.utils.multiclass import unique_labels

PADDED = 1
UNKNOWN = 0


def load_data(file, dev=False):
    """
    Parses and loads the training/dev/test data into a list of dicts

    :param dev:
    :param file:
    :return:
    """
    if dev:
        base_path = 'data/blurbs_dev_participants/'
    else:
        base_path = 'data/blurbs_test_participants/'
    full_path = join(base_path, file)

    labels_by_level = {'0': defaultdict(int),
                       '1': defaultdict(int),
                       '2': defaultdict(int)}

    with open(full_path, 'rt') as f_in:
        print("Loading {}".format(full_path))
        soup = BeautifulSoup(f_in, "html.parser")
        data_x = []
        data_y = []
        for book in soup.findAll("book"):
            x = {'title': book.title.text,
                 'body': book.body.text,
                 'authors': book.authors.text,
                 'published': book.published.text,
                 'isbn': book.isbn.text}

            if 'train' in full_path:
                categories = []
                for categ in book.categories:
                    if isinstance(categ, NavigableString):
                        continue
                    topics = {}
                    for t in categ:
                        if isinstance(t, Tag):
                            level = int(t['d'])
                            topics[level] = t.text
                            labels_by_level[str(level)][t.text] += 1
                    if topics is not None and len(topics) > 0:
                        categories.append(topics)
                data_y.append(categories)
            data_x.append(x)

        print(f'Loaded {len(data_x)} documents')

    return data_x, data_y, labels_by_level


def generate_submission_file(predictions, ml_binarizer, dev_data_x):
    """
    All submissions should be formatted as shown below (submissions to both tasks) and written
    into a file called answer.txt and uploaded as a zipped file:

    subtask_a
    ISBN<Tab>Label1<Tab>Label2<Tab>....<Tab>Label_n
    ISBN<Tab>Label1<Tab>Label2<Tab>....<Tab>Label_n...
    subtask_b
    ISBN<Tab>Label1<Tab>Label2<Tab>....<Tab>Label_n
    ISBN<Tab>Label1<Tab>Label2<Tab>....<Tab>Label_n...

    :return:
    """

    with open('answer.txt', 'wt') as f_out:
        f_out.write(str('subtask_a\n'))
        for pred, data in zip(ml_binarizer.inverse_transform(predictions), dev_data_x):
            f_out.write(data['isbn'] + '\t' + '\t'.join([p for p in pred]) + '\n')


def build_token_index(x_data, lower=False, simple=False):

    vocabulary = set()
    token_freq = Counter()
    stop_words = set(stopwords.words('german'))
    max_sent_length = 0

    if not simple:
        for x in x_data:
            text = x['title'] + " SEP " + x['body']
            sentences = sent_tokenize(text, language='german')
            for s in sentences:
                tmp_len = 0
                if lower is True:
                    words = [word.lower() for word in word_tokenize(s) if word not in stop_words]
                    vocabulary.update(words)
                    for token in words:
                        token_freq[token] += 1
                else:
                    words = word_tokenize(s)
                    vocabulary.update([word for word in word_tokenize(s) if word not in stop_words])
                    for token in words:
                        token_freq[token] += 1
                tmp_len += len(s)
                max_sent_length = max(tmp_len, max_sent_length)

        token2idx = {word: i + 2 for i, word in enumerate(vocabulary, 0)}
        token2idx["PADDED"] = PADDED
        token2idx["UNKNOWN"] = UNKNOWN

    elif simple is True:
        pass

    return token2idx, max_sent_length, token_freq


def vectorizer(x_sample, token2idx):
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


def vectorize_dev_data(dev_data_x, max_sent_len, token2idx):
    print("Vectorizing dev data\n")
    vectors = []
    for x in dev_data_x:
        tokens = []
        text = x['title'] + " SEP " + x['body']
        sentences = sent_tokenize(text, language='german')
        for s in sentences:
            tokens += word_tokenize(s)
        vector = vectorizer(tokens, token2idx)
        vectors.append(vector)
    test_vectors = pad_sequences(vectors, padding='post', maxlen=max_sent_len,
                                 truncating='post', value=token2idx['PADDED'])
    return test_vectors


def vectorize_one_sample(x, max_sent_len, token2idx):
    tokens = []
    text = x['title'] + " SEP " + x['body']
    sentences = sent_tokenize(text, language='german')
    for s in sentences:
        tokens += word_tokenize(s)
    vector = vectorizer(tokens, token2idx)
    padded_vector = pad_sequences([vector], padding='post', maxlen=max_sent_len,
                                  truncating='post', value=token2idx['PADDED'])
    return padded_vector


def plot_confusion_matrix(y_true, y_pred, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def plot_precision_recall_curve(class_name, y_test, y_score, timestamp):
    plt.figure()
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve {}'.format(class_name))
    plt.savefig(f'{class_name}_precision_recall_{timestamp}.png')

    return precision, recall


def write_reports_to_disk(all_preds, all_scores, all_true, model, classes, clf=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(" ", "_").replace(":", "_")

    # write classification report to file
    report = classification_report(all_true, all_preds)
    with open('classification_report_' + timestamp + '.txt', 'wt') as f_out:
        f_out.write(report)

    # write confusion matrices to file
    np.set_printoptions(precision=2)
    plt.rcParams['figure.figsize'] = [8, 6]
    plt.rcParams['font.size'] = 11

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(all_true, all_preds, title='Confusion matrix, without normalization')
    plt.savefig('confusion_matrix_' + timestamp + '.png')

    # Plot normalized confusion matrix
    plot_confusion_matrix(all_true, all_preds, normalize=True, title='Normalized confusion matrix')
    plt.savefig('confusion_matrix_normalized_' + timestamp + '.png')

    # precision-recall curves
    if len(classes) > 2:
        print(f'Computer precision/recall curves for {len(classes)} classes')
        if clf == 'bucket':
            array = np.array(all_scores)
            all_true_array = np.where(array > 0.5, 1, 0)
            all_scores_array = np.array(all_scores)
        else:
            all_true_array = to_categorical(model.le.transform(all_true))
            all_scores_array = np.array(all_scores)

        for i in range(len(classes)):
            precision, recall = plot_precision_recall_curve(classes[i],
                                                            all_true_array[:, i],
                                                            all_scores_array[:, i],
                                                            timestamp)

            # find all the indexes for precision values between 0.90 and 0.95
            boolean_1 = (precision >= 0.90)
            boolean_2 = (precision <= 0.96)
            result = np.where(np.logical_and(boolean_1, boolean_2))
            with open('precision_recall_values_' + timestamp + '.txt', 'a') as f_out:
                f_out.write(classes[i] + '\n')
                separator = len(classes[i]) * "-"
                f_out.write(separator + '\n')
                f_out.write("precision\trecall\n")
                for x in result[0]:
                    out = str(precision[x]) + '\t' + str(recall[x]) + '\n'
                    f_out.write(out)
                f_out.write('\n\n')

    else:
        print(f'Computer precision/recall curves for {len(classes)} classes')
        plt.figure()
        precision, recall, _ = precision_recall_curve(all_true, all_scores)

        # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
        #step_kwargs = ({'step': 'post'}
        #               if 'step' in signature(plt.fill_between).parameters
        #               else {})
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.savefig(f'precision_recall_{timestamp}.png')

        # find all the indexes for precision values between 0.90 and 0.95
        boolean_1 = (precision >= 0.90)
        boolean_2 = (precision <= 0.95)
        result = np.where(np.logical_and(boolean_1, boolean_2))
        with open('precision_recall_values_' + timestamp + '.txt', 'a') as f_out:
            # f_out.write(classes[i] + '\n')
            # separator = len(classes[i]) * "-"
            # f_out.write(separator + '\n')
            f_out.write("precision\trecall\n")
            for x in result[0]:
                out = str(precision[x]) + '\t' + str(recall[x]) + '\n'
                f_out.write(out)
            f_out.write('\n\n')