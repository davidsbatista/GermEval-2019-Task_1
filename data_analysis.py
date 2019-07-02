from utils.pre_processing import load_data
from utils.statistical_analysis import data_analysis


def main():
    # load train and test data
    train_data_x, train_data_y, labels = load_data('blurbs_train_all.txt')

    # data analysis
    # data_analysis(train_data_x, train_data_y, train_data_x)

    """
    # load dev data
    dev_data_x, dev_data_y, labels = load_data('blurbs_dev_participants.txt', dev=True)

    # get all ISBNS
    dev_isbns = {x['isbn'] for x in dev_data_x}

    # load all train data and select only ISBNS which are also in the dev_data
    train_data_all_x, train_data_all_y, labels = load_data('blurbs_train_all.txt', dev=False)

    dev_data_x = []
    dev_data_y = []

    for x,y in zip(train_data_all_x, train_data_all_y):
        if x['isbn'] in dev_isbns:
            dev_data_x.append(x)
            dev_data_y.append(y)
    print(len(dev_data_x))
    """

    # data analysis
    data_analysis(train_data_x, train_data_y, train_data_x)


if __name__ == '__main__':
    main()
