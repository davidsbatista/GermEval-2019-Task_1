from utils.pre_processing import load_data
from utils.statistical_analysis import data_analysis


def main():
    # load train and test data
    train_data_x, train_data_y, labels = load_data('blurbs_train_all.txt')

    # load dev data
    dev_data_x, dev_data_y, labels = load_data('blurbs_train_all.txt')

    # data analysis
    data_analysis(train_data_x, train_data_y, test_data_x)


if __name__ == '__main__':
    main()
