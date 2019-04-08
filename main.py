import numpy as np
import json

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from data_utils import Data
from models.char_cnn_kim import CharCNNKim
from utils import load_data


def main():
    # Load configurations
    config = json.load(open("config.json"))

    # load train data
    train_data_x, train_data_y, labels = load_data('blurbs_train.txt')

    # load dev data
    dev_data_x, _, _ = load_data('blurbs_dev_participants.txt')

    data_y_level_0 = []
    for y_labels in train_data_y:
        labels_0 = set()
        for label in y_labels:
            labels_0.add(label[0])
        data_y_level_0.append(list(labels_0))
    train_data_y = data_y_level_0

    ml_binarizer = MultiLabelBinarizer()
    y_labels = ml_binarizer.fit_transform(train_data_y)
    print('Total of {} classes'.format(len(ml_binarizer.classes_)))
    data_y = y_labels

    new_data_x = [x['title'] + " SEP " + x['body'] for x in train_data_x]

    # split into train and hold out set
    train_x, test_x, train_y, test_y = train_test_split(new_data_x, data_y,
                                                        random_state=42,
                                                        test_size=0.30)
    data_raw = []
    for x, y in zip(train_x[:100], train_y[:100]):
        data_raw.append((y, x))
    data = np.array(data_raw)
    training_data = Data(data_source='',
                         alphabet=config["data"]["alphabet"],
                         input_size=config["data"]["input_size"],
                         num_of_classes=config["data"]["num_of_classes"])
    # training_data.load_data()
    training_data.data = data
    training_inputs, training_labels = training_data.get_all_data()

    # Load validation data
    data_raw = []
    for x, y in zip(train_x[:50], train_y[:50]):
        data_raw.append((y, x))
    data = np.array(data_raw)
    validation_data = Data(data_source='',
                           alphabet=config["data"]["alphabet"],
                           input_size=config["data"]["input_size"],
                           num_of_classes=config["data"]["num_of_classes"])
    validation_data.data = data
    validation_inputs, validation_labels = validation_data.get_all_data()

    # Load model configurations and build model
    model = CharCNNKim(input_size=config["data"]["input_size"],
                       alphabet_size=config["data"]["alphabet_size"],
                       embedding_size=config["char_cnn_kim"]["embedding_size"],
                       conv_layers=config["char_cnn_kim"]["conv_layers"],
                       fully_connected_layers=config["char_cnn_kim"]["fully_connected_layers"],
                       num_of_classes=8,
                       dropout_p=config["char_cnn_kim"]["dropout_p"],
                       optimizer=config["char_cnn_kim"]["optimizer"],
                       loss=config["char_cnn_kim"]["loss"])

    # Train model
    model.train(training_inputs=training_inputs,
                training_labels=training_labels,
                validation_inputs=[],
                validation_labels=[],
                epochs=100,
                batch_size=config["training"]["batch_size"],
                checkpoint_every=config["training"]["checkpoint_every"])

    preds = model.test(testing_inputs=validation_inputs, testing_labels=validation_labels,
                       batch_size=32)

    preds = np.where(preds > 0.5, 1, 0)
    preds_labels = ml_binarizer.inverse_transform(preds)
    print(classification_report(train_y[:50], preds))


if __name__ == "__main__":
    main()
