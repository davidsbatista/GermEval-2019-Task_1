import datetime
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.utils.fixes import signature
from sklearn.utils.multiclass import unique_labels

stopwords = ['a',
'aber',
'about',
'above',
'after',
'again',
'against',
'ain',
'all',
'alle',
'allem',
'allen',
'aller',
'alles',
'als',
'also',
'am',
'an',
'and',
'ander',
'andere',
'anderem',
'anderen',
'anderer',
'anderes',
'anderm',
'andern',
'anderr',
'anders',
'any',
'are',
'aren',
'aren\'t',
'as',
'at',
'auch',
'auf',
'aus',
'be',
'because',
'been',
'before',
'bei',
'being',
'below',
'between',
'bin',
'bis',
'bist',
'both',
'but',
'by',
'can',
'couldn',
'couldn\'t',
'd',
'da',
'damit',
'dann',
'das',
'dasselbe',
'dazu',
'daß',
'dein',
'deine',
'deinem',
'deinen',
'deiner',
'deines',
'dem',
'demselben',
'den',
'denn',
'denselben',
'der',
'derer',
'derselbe',
'derselben',
'des',
'desselben',
'dessen',
'dich',
'did',
'didn',
'didn\'t',
'die',
'dies',
'diese',
'dieselbe',
'dieselben',
'diesem',
'diesen',
'dieser',
'dieses',
'dir',
'do',
'doch',
'does',
'doesn',
'doesn\'t',
'doing',
'don',
'don\'t',
'dort',
'down',
'du',
'durch',
'during',
'each',
'ein',
'eine',
'einem',
'einen',
'einer',
'eines',
'einig',
'einige',
'einigem',
'einigen',
'einiger',
'einiges',
'einmal',
'er',
'es',
'etwas',
'euch',
'euer',
'eure',
'eurem',
'euren',
'eurer',
'eures',
'few',
'for',
'from',
'further',
'für',
'gegen',
'gewesen',
'hab',
'habe',
'haben',
'had',
'hadn',
'hadn\'t',
'has',
'hasn',
'hasn\'t',
'hat',
'hatte',
'hatten',
'have',
'haven',
'haven\'t',
'having',
'he',
'her',
'here',
'hers',
'herself',
'hier',
'him',
'himself',
'hin',
'hinter',
'his',
'how',
'i',
'ich',
'if',
'ihm',
'ihn',
'ihnen',
'ihr',
'ihre',
'ihrem',
'ihren',
'ihrer',
'ihres',
'im',
'in',
'indem',
'ins',
'into',
'is',
'isn',
'isn\'t',
'ist',
'it',
'it\'s',
'its',
'itself',
'jede',
'jedem',
'jeden',
'jeder',
'jedes',
'jene',
'jenem',
'jenen',
'jener',
'jenes',
'jetzt',
'just',
'kann',
'kein',
'keine',
'keinem',
'keinen',
'keiner',
'keines',
'können',
'könnte',
'll',
'm',
'ma',
'machen',
'man',
'manche',
'manchem',
'manchen',
'mancher',
'manches',
'me',
'mein',
'meine',
'meinem',
'meinen',
'meiner',
'meines',
'mich',
'mightn',
'mightn\'t',
'mir',
'mit',
'more',
'most',
'muss',
'musste',
'mustn',
'mustn\'t',
'my',
'myself',
'nach',
'needn',
'needn\'t',
'nicht',
'nichts',
'no',
'noch',
'nor',
'not',
'now',
'nun',
'nur',
'o',
'ob',
'oder',
'of',
'off',
'ohne',
'on',
'once',
'only',
'or',
'other',
'our',
'ours',
'ourselves',
'out',
'over',
'own',
're',
's',
'same',
'sehr',
'sein',
'seine',
'seinem',
'seinen',
'seiner',
'seines',
'selbst',
'shan',
'shan\'t',
'she',
'she\'s',
'should',
'should\'ve',
'shouldn',
'shouldn\'t',
'sich',
'sie',
'sind',
'so',
'solche',
'solchem',
'solchen',
'solcher',
'solches',
'soll',
'sollte',
'some',
'sondern',
'sonst',
'such',
't',
'than',
'that',
'that\'ll',
'the',
'their',
'theirs',
'them',
'themselves',
'then',
'there',
'these',
'they',
'this',
'those',
'through',
'to',
'too',
'um',
'und',
'under',
'uns',
'unser',
'unsere',
'unserem',
'unseren',
'unseres',
'unter',
'until',
'up',
've',
'very',
'viel',
'vom',
'von',
'vor',
'war',
'waren',
'warst',
'was',
'wasn',
'wasn\'t',
'we',
'weg',
'weil',
'weiter',
'welche',
'welchem',
'welchen',
'welcher',
'welches',
'wenn',
'werde',
'werden',
'were',
'weren',
'weren\'t',
'what',
'when',
'where',
'which',
'while',
'who',
'whom',
'why',
'wie',
'wieder',
'will',
'wir',
'wird',
'wirst',
'with',
'wo',
'wollen',
'wollte',
'won',
'won\'t',
'wouldn',
'wouldn\'t',
'während',
'würde',
'würden',
'y',
'you',
'you\'d',
'you\'ll',
'you\'re',
'you\'ve',
'your',
'yours',
'yourself',
'yourselves',
'zu',
'zum',
'zur',
'zwar',
'zwischen', 'über']


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
        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
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