__author__ = 'fuadissa'

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow import nn


def build_graph(training_setting):
    graph = tf.Graph()

    with graph.as_default():
        with tf.name_scope('inputs') as name_scope:
            X = tf.placeholder(tf.int32, [None, training_setting['maximum_sent_length']], name='x')
            y = tf.placeholder(tf.float32, [None, training_setting['classes_num']], name='y')
            dropout = tf.placeholder(tf.float32, shape=[], name='dropout')

            pretrained_embeddings_input = tf.placeholder(tf.float32, shape=[training_setting['pretrained_vocab_length'],
                                                                            training_setting['embedding_size']],
                                                         name='pretrained_embeddings_ph')

        with tf.name_scope('embedding') as name_scope:
            reserved_embeddings = tf.Variable(tf.random_uniform([training_setting['reserved_vocab_length'],
                                                                 training_setting['embedding_size']], -1.0, 1.0),
                                              trainable=True, name='reserved_embeddings')


            if training_setting['use_pretrained_embeddings']:
                pretrained_embeddings = tf.Variable(tf.random_uniform([training_setting['pretrained_vocab_length'],
                                                                   training_setting['embedding_size']], -1.0, 1.0),
                                                        trainable=False, name='pretrained_embeddings')

                assign_pretrained_embeddings = tf.assign(pretrained_embeddings, pretrained_embeddings_input, name='assign_pretrained_embeddings')
            else:

                pretrained_embeddings = tf.Variable(tf.random_uniform([training_setting['pretrained_vocab_length'],
                                                                   training_setting['embedding_size']], -1.0, 1.0),
                                                    trainable=True, name='pretrained_embeddings')

            X = tf.where(tf.less(X, tf.constant(training_setting['reserved_vocab_length'])), X * 2,
                         (X - training_setting['reserved_vocab_length']) * 2 + 1)

            word_embeddings = tf.nn.embedding_lookup([reserved_embeddings, pretrained_embeddings], X, name='word_embeddings')

        with tf.name_scope('gru_cell') as name_scope:
            gru_forward = rnn.DropoutWrapper(rnn.GRUCell(training_setting['hidden_units']), output_keep_prob=dropout)
            gru_backward = rnn.DropoutWrapper(rnn.GRUCell(training_setting['hidden_units']), output_keep_prob=dropout)

            (gru_output_forward, gru_output_backward), _ = nn.bidirectional_dynamic_rnn(gru_forward, gru_backward, word_embeddings, dtype=tf.float32)
            bidirectional_gru_output = tf.concat(axis=2, values=(gru_output_forward, gru_output_backward), name='output')

        with tf.name_scope('pooling') as name_scope:
            W = tf.Variable(tf.random_normal([2 * training_setting['hidden_units']], name='attention_weight'))
            b = tf.Variable(tf.random_normal([1]), name='attention_bias')
            attentions = tf.reduce_sum(tf.multiply(W, bidirectional_gru_output), axis=2) + b
            attentions = tf.nn.softmax(attentions)

            expand_attentions = tf.expand_dims(attentions, 1)
            transpose_outputs = tf.transpose(bidirectional_gru_output, perm=[0, 2 ,1])

            attentions_output = tf.reduce_sum(tf.transpose(tf.multiply(expand_attentions, transpose_outputs),
                                                           perm=[0, 2, 1]), axis=1)

        with tf.name_scope('mlp') as name_scope:
            W_mlp = tf.Variable(tf.random_normal([2 * training_setting['hidden_units'], training_setting['classes_num']]))
            b_mlp = tf.Variable(tf.random_normal([training_setting['classes_num']]))

            logits = tf.matmul(attentions_output, W_mlp) + b_mlp
            probability = tf.nn.softmax(logits, name='probability')
            y_pred = tf.argmax(probability, 1, name='y_pred')

        with tf.name_scope('loss') as name_scope:
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y, name='loss')

        with tf.name_scope('optimizer') as name_scope:
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=training_setting['learning_rate']).minimize(loss, name='optimizer')

    return graph
