__author__ = 'fuadissa'

import pandas as pd


def get_dataset(file_path, limit=None):
    dataset_df = pd.read_pickle(file_path)

    if limit:
        dataset_df = dataset_df.head(limit)

    dataset_df['accepted'] = True
    dataset_df.loc[dataset_df['gold_label'].apply(lambda x: x not in set(['contradiction', 'neutral', 'entailment'])),'accepted'] = False
    dataset_df.drop(dataset_df[dataset_df['accepted'] == False].index, inplace=True)

    x_sent_1 = dataset_df['sentence1_tokenized'].tolist()
    x_sent_2 = dataset_df['sentence2_tokenized'].tolist()

    y = dataset_df['gold_label'].tolist()

    return x_sent_1, x_sent_2, y

def batcher(lists_to_batch, batch_size, infinite=False):
    length_of_list = len(lists_to_batch[0])
    start_index = 0
    while True:
        new_start = False
        batched_lists = list()
        if start_index + batch_size < length_of_list:
            end_index = start_index + batch_size
        else:
            end_index = length_of_list
            new_start = True

        for list_to_batch in lists_to_batch:
            batched_lists.append(list_to_batch[start_index: end_index])

        if new_start:
            start_index = 0
            if not infinite:
                break
        else:
            start_index += batch_size

        yield tuple(batched_lists), new_start
    yield tuple(batched_lists), new_start