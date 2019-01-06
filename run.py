__author__ = 'fuadissa'

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer

from processing.vocab_processor import VocabProcessor
from processing.input_processing import get_dataset, batcher
from model.train_model import train_network
from settings import PROCESSED_DATA_FILE, LOGGER, PARTIAL_PROCESSED_DATA_FILE

training_settings = {
    'model_path': './trained_models/test/',
    'use_pretrained_embeddings': True,
    'embedding_size': 300,
    'batch_size': 512,
    'hidden_units': 32,
    'learning_rate': 0.01,
    'patience': 1024000,
    'train_interval': 1,
    'valid_interval': 1,
    'max_epoch': 1000,
    'dropout':  0.7,
    'maximum_sent_length': None,
    'classes_num': None,
    'reserved_vocab_length': None,
    'pretrained_vocab_length': None,
}

def train(training_settings):

    X_sent1, X_sent2, y = get_dataset(PROCESSED_DATA_FILE, limit=10000)
    LOGGER.info('Got dataset')

    all_sentences = X_sent1 + X_sent2
    label_encoder = LabelBinarizer()
    y = label_encoder.fit_transform(y)

    vocab = [each for each_sent in all_sentences for each in each_sent]
    vocab_processor = VocabProcessor(vocab)
    embedding = vocab_processor.get_word2vec()
    training_settings['reserved_vocab_length'] = len(vocab_processor.embedding_vocab)
    training_settings['pretrained_vocab_length'] = len(vocab_processor.reserved_vocab)

    X_sent1_transformed = vocab_processor.transform(X_sent1)

    padding_size = max([len(each) for each in X_sent1_transformed])
    X_sent1_padded = vocab_processor.pad(X_sent1_transformed, padding_size)

    X_sent2_transformed = vocab_processor.transform(X_sent2)
    padding_size = max([len(each) for each in X_sent2_transformed])
    X_sent2_padded = vocab_processor.pad(X_sent2_transformed, padding_size)

    X_sent1_train, X_sent1_test, X_sent2_train, X_sent2_test, y_train, y_test = \
        train_test_split(X_sent1_padded, X_sent2_padded, y, test_size=0.2, random_state=0)

    train_batcher = batcher([X_sent1_train, X_sent2_train, y_train], training_settings['batch_size'], infinite=True)
    valid_batcher = batcher([X_sent1_test, X_sent2_test, y_test], training_settings['batch_size'])
    train_number_of_instance = len(X_sent1_train)
    training_settings['classes_num'] = len(label_encoder.classes_)

    LOGGER.info('Number of training instances : {}'.format(train_number_of_instance))
    train_network(training_settings,
                  train_batcher,
                  list(valid_batcher),
                  embedding,
                  train_number_of_instance)

if __name__ == '__main__':
    train(training_settings)