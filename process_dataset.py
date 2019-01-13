__author__ = 'fuadissa'

import pandas as pd
from nltk.tokenize import word_tokenize

from settings import TRAIN_DATA_FILE, PROCESSED_DATA_FILE, TEST_DATA_FILE, DEV_DATA_FILE, PARTIAL_PROCESSED_DATA_FILE
from settings import LOGGER

def word_tokenize_list(sentences_list):
    tokenized_list = list()
    total = len(sentences_list)
    for index , each in enumerate(sentences_list):
        tokenized_list.append(word_tokenize(each))
        LOGGER.info('tokenized : {}/{}'.format(index, total))

    return tokenized_list

if __name__ == '__main__':
    dataset_df_train = pd.read_json(TRAIN_DATA_FILE, lines=True)
    dataset_df_dev = pd.read_json(DEV_DATA_FILE, lines=True)
    dataset_df_test = pd.read_json(TEST_DATA_FILE, lines=True)
    dataset_df_train['type'] = 'train'
    dataset_df_test['type'] = 'test'
    dataset_df_dev['type'] = 'dev'

    dataset_df = dataset_df_train.append(dataset_df_dev, ignore_index=True)
    dataset_df = dataset_df.append(dataset_df_test, ignore_index=True)

    sentence_1 = dataset_df['sentence1'].tolist()
    sentence_2 = dataset_df['sentence2'].tolist()

    dataset_df['sentence1_tokenized'] = word_tokenize_list(sentence_1)
    dataset_df['sentence2_tokenized'] = word_tokenize_list(sentence_2)

    dataset_df.to_pickle(PROCESSED_DATA_FILE)