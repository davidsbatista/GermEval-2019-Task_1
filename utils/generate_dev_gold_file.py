from .pre_processing import load_data


def main():

    # load dev data
    dev_data_x, _, _ = load_data('blurbs_dev_participants.txt', dev=True)
    dev_gold = [{'isbn': x['isbn'], 'gold': None} for x in dev_data_x]

    # load all train data and get the labels for the ISBNS in dev_data
    train_data_all_x, train_data_all_y, labels = load_data('blurbs_train_all.txt', dev=False)

    c = 0
    with open('gold.txt', 'wt') as f_out:
        f_out.write(str('subtask_a\n'))
        for x, y in zip(train_data_all_x, train_data_all_y):
            all_labels = set([label[0] for label in y])
            if x['isbn'] in [x['isbn'] for x in dev_gold]:
                f_out.write(x['isbn'] + '\t' + '\t'.join(all_labels) + '\n')

        f_out.write(str('subtask_b\n'))
        for x, y in zip(train_data_all_x, train_data_all_y):
            all_labels = set([x for label in y for x in label.values()])
            if x['isbn'] in [x['isbn'] for x in dev_gold]:
                f_out.write(x['isbn'] + '\t' + '\t'.join(all_labels) + '\n')

        c += 1


if __name__ == '__main__':
    main()
