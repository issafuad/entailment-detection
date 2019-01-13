__author__ = 'fuadissa'

import pandas as pd


def get_dataset(file_path, limit=None):
    def get_X_and_y(dataframe):
        x_sent1 = dataframe['sentence1_tokenized'].tolist()
        x_sent2 = dataframe['sentence2_tokenized'].tolist()
        y = dataframe['gold_label'].tolist()
        return X_sent1, X_sent2, y

    dataset_df = pd.read_pickle(file_path)

    if limit:
        dataset_df = dataset_df.head(limit)

    dataset_df['accepted'] = True
    dataset_df.loc[dataset_df['gold_label'].apply(lambda x: x not in set(['contradiction', 'neutral', 'entailment'])),'accepted'] = False
    dataset_df.drop(dataset_df[dataset_df['accepted'] == False].index, inplace=True)

    dataset_df_train = dataset_df[dataset_df['type'] == 'train']
    dataset_df_dev = dataset_df[dataset_df['type'] == 'dev']
    dataset_df_test = dataset_df[dataset_df['type'] == 'test']

    X_sent1_train, X_sent2_train, y_train = get_X_and_y(dataset_df_train)
    X_sent1_dev, X_sent2_dev, y_dev = get_X_and_y(dataset_df_dev)
    X_sent1_test, X_sent2_test, y_test = get_X_and_y(dataset_df_test)

    return X_sent1_train, X_sent2_train, y_train, X_sent1_dev, X_sent2_dev, y_dev, X_sent1_test, X_sent2_test, y_test

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