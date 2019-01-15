__author__ = 'fuadissa'

import numpy as np

from settings import WORD2VEC
from gensim.models import KeyedVectors
from settings import LOGGER


class VocabProcessor(object):
    SENT_START = '__START__'
    SENT_END = '__END__'
    UNKNOWN = '__UNKNOWN__'
    PADDING = '__PADDED__'
    SENT_SEPARATOR = '__SEP__'

    PROCESSOR_RESERVED = [SENT_START, SENT_END, UNKNOWN, PADDING, SENT_SEPARATOR]

    def __init__(self, vocab, reserved_vocab=list()):
        self.vocab_set = set(vocab)

        self.reserved_vocab = self.PROCESSOR_RESERVED + reserved_vocab
        self.reserved_vocab_set = set(self.reserved_vocab)

        self.embedding_vocab = list()
        self.all_vocab = list()
        self.vocab2id = dict()

    def get_word2vec(self):
        LOGGER.info('Loading Word2Vec Vocabulary')
        word2vec = KeyedVectors.load_word2vec_format(WORD2VEC, binary=True)
        LOGGER.info('Finished loading Word2Vec Vocabulary. {} loaded'.format(word2vec.vectors.shape[0]))
        chosen_word_index_set = set()
        for index, each_word in enumerate(word2vec.index2word):

            if not index % 100000:
                LOGGER.info('checked {}/{}'.format(index, len(word2vec.index2word)))

            if each_word in self.vocab_set and each_word not in self.reserved_vocab_set:
                chosen_word_index_set.add(index)

        LOGGER.info('Found {}/{} of dataset vocab'.format(len(chosen_word_index_set), len(self.vocab_set)))
        chosen_word_index_list = list(chosen_word_index_set)
        self.embedding_vocab = [word2vec.index2word[each] for each in chosen_word_index_list]
        self.all_vocab = self.reserved_vocab + self.embedding_vocab
        self.vocab2id = {vocab: index for index, vocab in enumerate(self.all_vocab)}
        word_embeddings = word2vec.vectors[chosen_word_index_list]
        reserved_embeddings = np.random.random((len(self.reserved_vocab), word_embeddings.shape[1]))
        word_embeddings_all = np.concatenate((reserved_embeddings, word_embeddings), axis=0)

        return word_embeddings_all

    def transform(self, X_sent1, X_sent2):
        transformed_X = list()

        for each_sent1, each_sent2 in zip(X_sent1, X_sent2):

            transformed_words = list()

            transformed_words.append(self.vocab2id[self.SENT_START])
            for each_word in each_sent1:
                transformed_words.append(
                    self.vocab2id.get(each_word, self.vocab2id.get(each_word.lower(), self.vocab2id.get(self.UNKNOWN))))
            transformed_words.append(self.vocab2id[self.SENT_END])

            transformed_words.append(self.vocab2id[self.SENT_SEPARATOR])

            transformed_words.append(self.vocab2id[self.SENT_START])
            for each_word in each_sent2:
                transformed_words.append(
                    self.vocab2id.get(each_word, self.vocab2id.get(each_word.lower(), self.vocab2id.get(self.UNKNOWN))))
            transformed_words.append(self.vocab2id[self.SENT_END])

            transformed_X.append(transformed_words)

        return transformed_X

    def pad(self, X, padding_size):

        padded_X = list()
        sentences_cut_counter = 0
        for each_sent in X:
            if len(each_sent) <= padding_size:
                padded_sent = each_sent + [self.vocab2id[self.PADDING]] * (padding_size - len(each_sent))
            else:
                padded_sent = each_sent[:padding_size]
                sentences_cut_counter += 1

            padded_X.append(padded_sent)

        LOGGER.info('number of shortened sentences : {}'.format(sentences_cut_counter))

        return padded_X
