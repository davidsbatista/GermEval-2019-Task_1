from collections import defaultdict
from os.path import join

from bs4 import BeautifulSoup, Tag, NavigableString


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
