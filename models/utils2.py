import datetime
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.utils.fixes import signature
from sklearn.utils.multiclass import unique_labels


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