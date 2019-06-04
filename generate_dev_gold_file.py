from utils import load_data


def main():

    # load dev data
    dev_data_x, _, _ = load_data('blurbs_dev_participants.txt', dev=True)
    dev_gold = [{'isbn': x['isbn'], 'gold': None} for x in dev_data_x]

    # load all train data and get the labels for the ISBNS in dev_data
    train_data_all_x, train_data_all_y, labels = load_data('blurbs_train_all.txt', dev=False)

    c = 0
    for x, y in zip(train_data_all_x, train_data_all_y):
        if x['isbn'] in [x['isbn'] for x in dev_gold]:
            print(x['isbn'])
            print(y)
            print()
            c += 1

    print(c)
    print()

    """
    with open('answer.txt', 'wt') as f_out:
        f_out.write(str('subtask_a\n'))
        for x in dev_data_x:
            isbn = x['isbn']
            f_out.write(str(isbn) + '\t' + '\t'.join(classification[isbn][0]) + '\n')

        f_out.write(str('subtask_b\n'))
        for x in dev_data_x:
            isbn = x['isbn']
            output = isbn + '\t'
            output += '\t'.join(classification[isbn][0])
            if len(classification[isbn][1]) > 0:
                output += '\t'
                output += '\t'.join(classification[isbn][1])
                if len(classification[isbn][2]) > 0:
                    output += '\t'
                    output += '\t'.join(classification[isbn][2])
            output += '\n'
            f_out.write(output)
    """


if __name__ == '__main__':
    main()
