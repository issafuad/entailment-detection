__author__ = 'fuadissa'

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
    'batch_size': 128,
    'hidden_units': 32,
    'learning_rate': 1,
    'patience': 10240000,
    'train_interval': 5,
    'valid_interval': 2,
    'max_epoch': 100,
    'dropout':  0.7,
    'maximum_sent_length': None,
    'classes_num': None,
    'reserved_vocab_length': None,
    'pretrained_vocab_length': None,
}

def train(training_settings):

    X_sent1_train, X_sent2_train, y_train,\
        X_sent1_dev, X_sent2_dev, y_dev,\
        X_sent1_test, X_sent2_test, y_test = get_dataset(PROCESSED_DATA_FILE)
    LOGGER.info('Loaded dataset')

    all_sentences = X_sent1_train + X_sent2_train + X_sent1_dev + X_sent2_dev
    label_encoder = LabelBinarizer()
    label_encoder.fit(y_train + y_dev + y_test)

    y_train = label_encoder.transform(y_train)
    y_dev = label_encoder.transform(y_dev)
    #y_test = label_encoder.transform(y_test)

    vocab = [each for each_sent in all_sentences for each in each_sent]
    vocab_processor = VocabProcessor(vocab)
    embedding = vocab_processor.get_word2vec()
    training_settings['reserved_vocab_length'] = len(vocab_processor.embedding_vocab)
    training_settings['pretrained_vocab_length'] = len(vocab_processor.reserved_vocab)

    X_sent_transformed_train = vocab_processor.transform(X_sent1_train, X_sent2_train)
    X_sent_transformed_dev = vocab_processor.transform(X_sent1_dev, X_sent2_dev)

    padding_size = max([len(each) for each in X_sent_transformed_train + X_sent_transformed_dev])
    X_sent_padded_train = vocab_processor.pad(X_sent_transformed_train, padding_size)
    X_sent_padded_dev = vocab_processor.pad(X_sent_transformed_dev, padding_size)

    X_sent_train_input, y_train = shuffle(X_sent_padded_train, y_train)
    X_sent_dev_input, y_dev = shuffle(X_sent_padded_dev, y_dev)

    train_batcher = batcher([X_sent_train_input, y_train], training_settings['batch_size'], infinite=True)
    valid_batcher = batcher([X_sent_dev_input, y_dev], training_settings['batch_size'])
    train_number_of_instance = len(X_sent_train_input)

    training_settings['classes_num'] = len(label_encoder.classes_)

    LOGGER.info('Number of training instances : {}'.format(train_number_of_instance))
    train_network(training_settings,
                  train_batcher,
                  list(valid_batcher),
                  embedding,
                  train_number_of_instance)

if __name__ == '__main__':
    train(training_settings)