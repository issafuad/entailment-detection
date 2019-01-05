__author__ = 'fuadissa'


training_settings = {
    'model_path': None,
    'use_pretrained_embeddings': True,
    'reserved_vocab_length' : None,
    'pretrained_vocab_length' : None,
    'embedding_size' : 300,
    'batch_size' : 32,
    'maximum_sent_length': None,
    'classes_num': None,
    'hidden_units': 64,
    'learning_rate' : 0.001,
    'patience': 1024000,
    'train_interval': 10,
    'valid_interval': 2,
    'max_epoch': 10,
    'dropout':  0.7
}