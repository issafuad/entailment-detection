__author__ = 'fuadissa'

import os
import logging


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(level=logging.INFO)

LOGGER.addHandler(ch)

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))

TRAINED_MODELS_PATH = os.path.join(PROJECT_ROOT, 'trained_models')
DATA_FILE_PATH = os.path.join(PROJECT_ROOT, 'data')
TEST_DATA_FILE = os.path.join(DATA_FILE_PATH, 'snli_1.0_test.json')
TRAIN_DATA_FILE = os.path.join(DATA_FILE_PATH, 'snli_1.0_train.json')
DEV_DATA_FILE = os.path.join(DATA_FILE_PATH, 'snli_1.0_dev.json')
PROCESSED_DATA_FILE = os.path.join(DATA_FILE_PATH, 'snli_processed.pkl')
WORD2VEC = os.path.join(DATA_FILE_PATH, 'GoogleNews-vectors-negative300.bin')


