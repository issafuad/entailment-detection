__author__ = 'fuadissa'

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
import argparse

from processing.vocab_processor import VocabProcessor
from processing.input_processing import get_dataset, batcher
from model.train_model import train_network
from settings import PROCESSED_DATA_FILE, LOGGER


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train(training_settings):
    LOGGER.info('Start Training')

    X_sent1_train, X_sent2_train, y_train, \
    X_sent1_dev, X_sent2_dev, y_dev, \
    X_sent1_test, X_sent2_test, y_test = get_dataset(PROCESSED_DATA_FILE)
    LOGGER.info('Loaded dataset')

    all_sentences = X_sent1_train + X_sent2_train + X_sent1_dev + X_sent2_dev
    label_encoder = LabelBinarizer()
    label_encoder.fit(y_train + y_dev + y_test)

    y_train = label_encoder.transform(y_train)
    y_dev = label_encoder.transform(y_dev)
    # y_test = label_encoder.transform(y_test)

    vocab = [each for each_sent in all_sentences for each in each_sent]
    vocab_processor = VocabProcessor(vocab)
    embedding = vocab_processor.get_word2vec()
    training_settings['reserved_vocab_length'] = len(vocab_processor.embedding_vocab)
    training_settings['pretrained_vocab_length'] = len(vocab_processor.reserved_vocab)

    X_sent_transformed_train = vocab_processor.transform(X_sent1_train, X_sent2_train)
    X_sent_transformed_dev = vocab_processor.transform(X_sent1_dev, X_sent2_dev)

    padding_size = max([len(each) for each in X_sent_transformed_train + X_sent_transformed_dev])
    training_settings['maximum_sent_length'] = padding_size
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


def get_arguments():
    parser = argparse.ArgumentParser(description='Parameters of the model')
    parser.add_argument('model_path', type=str)
    parser.add_argument('--use_pretrained_embeddings', nargs='?', type=str2bool, default=True)
    parser.add_argument('--embedding_size', nargs='?', type=int, default=300)
    parser.add_argument('--batch_size', nargs='?', type=int, default=128)
    parser.add_argument('--hidden_units', nargs='?', type=int, default=32)
    parser.add_argument('--learning_rate', nargs='?', type=float, default=1.0)
    parser.add_argument('--patience', nargs='?', type=int, default=10240000)
    parser.add_argument('--train_interval', nargs='?', type=int, default=5)
    parser.add_argument('--valid_interval', nargs='?', type=int, default=2)
    parser.add_argument('--dropout', nargs='?', type=float, default=0.7)
    parser.add_argument('--max_epoch', nargs='?', type=int, default=200)
    return vars(parser.parse_args())

if __name__ == '__main__':
    training_settings = get_arguments()
    train(training_settings)
