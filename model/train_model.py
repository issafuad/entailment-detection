__author__ = 'fuadissa'

import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf

from model.model_graph import build_graph
from settings import TRAINED_MODELS_PATH, LOGGER

MODEL_PATH = os.path.join(TRAINED_MODELS_PATH)
CHECKPOINT = 'ckpt'
TENSORBOARD_FOLDER = 'tb'


def train_network(training_setting, train_batcher, valid_batcher, embedding, train_number_of_instances):
    def add_metric_summaries(mode, iteration, name2metric):
        """Add summary for metric."""
        metric_summary = tf.Summary()
        for name, metric in name2metric.items():
            metric_summary.value.add(tag='{}_{}'.format(mode, name), simple_value=metric)
        summary_writer.add_summary(metric_summary, global_step=iteration)

    def show_train_stats(epoch, iteration, losses, y_true, y_pred):
        # compute mean statistics
        loss = np.mean(losses)
        accuracy = accuracy_score(y_true, y_pred)
        LOGGER.info('Epoch={}, Iter={:,}, Mean Training Loss={:.4f}, Accuracy={:.4f}, '.format(epoch, iteration, loss,
                                                                                               accuracy))
        add_metric_summaries('train', iteration, {'cross_entropy': loss, 'accuracy': accuracy})
        LOGGER.info('\n{}'.format(classification_report(y_true, y_pred, digits=3)))

    def validate(epoch, iteration, batcher, best_loss, patience):
        """Validate the model on validation set."""

        losses, y_true, y_pred = list(), list(), list()
        for (X_batch_sent, y_true_batch), _ in batcher:
            y_pred_batch, loss_batch = session.run(
                [graph.get_tensor_by_name('mlp/y_pred:0'),
                 graph.get_tensor_by_name('loss/loss:0')],
                feed_dict={
                    'inputs/x_sent:0': X_batch_sent,
                    'inputs/y:0': y_true_batch,
                    'inputs/dropout:0': 1
                })
            losses.extend(loss_batch.tolist())
            y_pred.extend(np.argmax(y_pred_batch, axis=1))
            y_true.extend(np.argmax(y_true_batch, axis=1))

        # compute mean statistics
        loss = np.mean(losses)
        accuracy = accuracy_score(y_true, y_pred)

        LOGGER.info(
            'Epoch={}, Iter={:,}, Validation Loss={:.4f}, Accuracy={:.4f}'.format(epoch, iteration, loss, accuracy))
        add_metric_summaries('valid', iteration, {'cross_entropy': loss, 'validation_accuracy': accuracy})
        LOGGER.info('\n{}'.format(classification_report(y_true, y_pred, digits=3)))

        if loss < best_loss:
            LOGGER.info('Best score Loss so far, save the model.')
            save()
            best_loss = loss

            if iteration * 2 > patience:
                patience = iteration * 2
                LOGGER.info('Increased patience to {:,}'.format(patience))
        return best_loss, patience

    def save():
        saver.save(session, os.path.join(training_setting['model_path'], CHECKPOINT))
        LOGGER.info('Finished Saving')

    graph = build_graph(training_setting)
    pretrained_embeddings = embedding[training_setting['reserved_vocab_length']:]
    patience = training_setting['patience']
    best_valid_loss = np.float64('inf')
    with tf.Session(graph=graph) as session:

        summary_writer = tf.summary.FileWriter(os.path.join(training_setting['model_path'], TENSORBOARD_FOLDER),
                                               session.graph)
        saver = tf.train.Saver(name='saver')

        session.run(tf.global_variables_initializer())
        session.run(graph.get_operation_by_name('embedding/assign_pretrained_embeddings'),
                    feed_dict={'inputs/pretrained_embeddings_ph:0': pretrained_embeddings})

        batches_in_train = train_number_of_instances / training_setting['batch_size']
        train_stat_interval = max(batches_in_train // training_setting['train_interval'], 1)
        valid_stat_interval = max(batches_in_train // training_setting['valid_interval'], 1)

        y_true = list()
        y_pred = list()
        losses = list()
        for batch_num, ((X_batch_sent, y_true_batch), new_start) in enumerate(train_batcher):
            iteration = batch_num * training_setting['batch_size']
            epoch = 1 + iteration // train_number_of_instances
            if new_start:
                y_true = list()
                y_pred = list()
                losses = list()

            if iteration > train_number_of_instances * training_setting['max_epoch']:
                LOGGER.info('reached max epoch')
                break

            _, y_pred_batch, loss = session.run(
                [graph.get_operation_by_name('optimizer/optimizer'),
                 graph.get_tensor_by_name('mlp/y_pred:0'),
                 graph.get_tensor_by_name('loss/loss:0')],
                feed_dict={
                    'inputs/x_sent:0': X_batch_sent,
                    'inputs/y:0': y_true_batch,
                    'inputs/dropout:0': training_setting['dropout']
                }
            )

            y_pred.extend(np.argmax(y_pred_batch, axis=1))
            y_true.extend(np.argmax(y_true_batch, axis=1))
            losses.extend(loss.tolist())

            if batch_num % train_stat_interval == 0:
                show_train_stats(epoch, iteration, losses, y_true, y_pred)

            if batch_num % valid_stat_interval == 0:
                best_valid_loss, patience = validate(epoch, iteration, valid_batcher, best_valid_loss, patience)

            if iteration > patience:
                LOGGER.info('Iteration is more than patience, finish training.')
                break

        LOGGER.info('Finished fitting the model.')
        LOGGER.info('Best Validation Cross-entropy Loss: {:.4f}'.format(best_valid_loss))
