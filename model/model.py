__author__ = 'fuadissa'

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow import nn


def build_graph(training_setting):
    tf.reset_default_graph()
    graph = tf.Graph()


    with graph.as_default():
        with tf.name_scope('inputs') as name_scope:
            X_sent1 = tf.placeholder(tf.int32, [None, training_setting['maximum_sent_length']], name='x_sent1')
            X_sent2 = tf.placeholder(tf.int32, [None, training_setting['maximum_sent_length']], name='x_sent2')
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

                assign_pretrained_embeddings = tf.assign(pretrained_embeddings, pretrained_embeddings_input,
                                                         name='assign_pretrained_embeddings')
            else:

                pretrained_embeddings = tf.Variable(tf.random_uniform([training_setting['pretrained_vocab_length'],
                                                                       training_setting['embedding_size']], -1.0, 1.0),
                                                    trainable=True, name='pretrained_embeddings')

            X_sent1 = tf.where(tf.less(X_sent1, tf.constant(training_setting['reserved_vocab_length'])), X_sent1 * 2,
                         (X_sent1 - training_setting['reserved_vocab_length']) * 2 + 1)
            X_sent2 = tf.where(tf.less(X_sent2, tf.constant(training_setting['reserved_vocab_length'])), X_sent2 * 2,
                         (X_sent2 - training_setting['reserved_vocab_length']) * 2 + 1)

            word_embeddings_sent1 = tf.nn.embedding_lookup([reserved_embeddings, pretrained_embeddings], X_sent1,
                                                     name='word_embeddings_sent1')
            word_embeddings_sent2 = tf.nn.embedding_lookup([reserved_embeddings, pretrained_embeddings], X_sent2,
                                                     name='word_embeddings_sent1')

        with tf.name_scope('gru_cell_sent1') as name_scope:
            gru_forward_sent1 = rnn.DropoutWrapper(rnn.GRUCell(training_setting['hidden_units']), output_keep_prob=dropout)
            gru_backward_sent1 = rnn.DropoutWrapper(rnn.GRUCell(training_setting['hidden_units']), output_keep_prob=dropout)

            (gru_output_forward, gru_output_backward), _ = nn.bidirectional_dynamic_rnn(gru_forward_sent1, gru_backward_sent1,
                                                                                        word_embeddings_sent1,
                                                                                        dtype=tf.float32,
                                                                                        scope=name_scope)
            bidirectional_gru_output_sent1 = tf.concat(axis=2, values=(gru_output_forward, gru_output_backward),
                                                       name='output_sent1')

        with tf.name_scope('gru_cell_sent2') as name_scope:
            gru_forward_sent2 = rnn.DropoutWrapper(rnn.GRUCell(training_setting['hidden_units']), output_keep_prob=dropout)
            gru_backward_sent2 = rnn.DropoutWrapper(rnn.GRUCell(training_setting['hidden_units']), output_keep_prob=dropout)

            (gru_output_forward, gru_output_backward), _ = nn.bidirectional_dynamic_rnn(gru_forward_sent2, gru_backward_sent2,
                                                                                        word_embeddings_sent2,
                                                                                        dtype=tf.float32,
                                                                                        scope=name_scope)
            bidirectional_gru_output_sent2 = tf.concat(axis=2,
                                                       values=(gru_output_forward, gru_output_backward),
                                                       name='output_sent2')

        with tf.name_scope('concat') as name_scope:
            both_sentences_concat = tf.concat(axis=1,
                                              values=(bidirectional_gru_output_sent1, bidirectional_gru_output_sent2),
                                              name='output_sent1')
            #both_sentences_concat = tf.Print(both_sentences_concat, [both_sentences_concat], message='both_sentences_concat')

        with tf.name_scope('pooling') as name_scope:
            W = tf.Variable(tf.random_normal([2 * training_setting['hidden_units']], name='attention_weight'))
            b = tf.Variable(tf.random_normal([1]), name='attention_bias')
            attentions = tf.reduce_sum(tf.multiply(W, both_sentences_concat), axis=2) + b
            attentions = tf.nn.softmax(attentions)

            expand_attentions = tf.expand_dims(attentions, 1)
            transpose_outputs = tf.transpose(both_sentences_concat, perm=[0, 2, 1])

            attentions_output = tf.reduce_sum(tf.transpose(tf.multiply(expand_attentions, transpose_outputs),
                                                           perm=[0, 2, 1]), axis=1)

        with tf.name_scope('mlp') as name_scope:
            W_mlp = tf.Variable(
                tf.random_normal([2 * training_setting['hidden_units'], training_setting['classes_num']]))
            b_mlp = tf.Variable(tf.random_normal([training_setting['classes_num']]))

            logits = tf.matmul(attentions_output, W_mlp) + b_mlp
            probability = tf.nn.softmax(logits, name='probability')
            y_pred = tf.one_hot(tf.argmax(probability, 1), depth=training_setting['classes_num'], name='y_pred')
            y_pred = tf.Print(y_pred, [y_pred], message='y_pred')


        with tf.name_scope('loss') as name_scope:
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y, name='loss')

        with tf.name_scope('optimizer') as name_scope:
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=training_setting['learning_rate']).minimize(loss,
                                                                                                             name='optimizer')

    return graph
