__author__ = 'fuadissa'

import pandas as pd


def get_dataset(file_path, limit=None):
    dataset_df = pd.read_pickle(file_path)

    if limit:
        dataset_df = dataset_df.head(limit)

    x_sent_1 = dataset_df['sentence1_tokenized'].tolist()
    x_sent_2 = dataset_df['sentence2_tokenized'].tolist()

    y = dataset_df['gold_label'].tolist()

    return x_sent_1, x_sent_2, y

def batcher(lists_to_batch, batch_size, infinite=False):
    batched_lists = list()
    while True:
        for start_index in range(0, len(lists_to_batch[0]), batch_size):
            for list_to_batch in lists_to_batch:
                batched_lists.append(list_to_batch[start_index: min(start_index + batch_size, len(list_to_batch))])
            yield tuple(batched_lists)
        if infinite:
            break
