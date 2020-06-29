import argparse

import os
import itertools
import pickle

import pandas as pd

from bert_concept_embeddings.custom_layers import get_custom_objects
from bert_concept_embeddings.bert_data_generator import ConceptTokenizer, BatchGenerator, NegativeSamplingBatchGenerator
from bert_concept_embeddings.model import time_attention_cbow_model, time_attention_cbow_negative_sampling_model
from bert_concept_embeddings.utils import CosineLRSchedule

import tensorflow as tf


def process_raw_input(_raw_input_data_path, _training_data_path):
    """
    Process the raw input data
    :param _raw_input_data_path:
    :param _training_data_path:
    :return: save and return the training dataset
    """
    if os.path.exists(_training_data_path):
        _training_data = pd.read_pickle(_training_data_path)
    else:
        patient_event_sequence = pd.read_parquet(_raw_input_data_path)
        patient_event_sequence['concept_id_visit_orders'] = patient_event_sequence['concept_ids'].apply(len) * \
                                                            patient_event_sequence['visit_rank_order'].apply(
                                                                lambda v: [v])

        patient_concept_ids = patient_event_sequence.sort_values(['person_id', 'visit_rank_order']) \
            .groupby('person_id')['concept_ids'].apply(lambda x: list(itertools.chain(*x))).reset_index()

        patient_visit_ids = patient_event_sequence.sort_values(['person_id', 'visit_rank_order']) \
            .groupby('person_id')['concept_id_visit_orders'].apply(lambda x: list(itertools.chain(*x))).reset_index()

        event_dates = patient_event_sequence.sort_values(['person_id', 'visit_rank_order']) \
            .groupby('person_id')['dates'].apply(lambda x: list(itertools.chain(*x))).reset_index()
        
        event_periods = patient_event_sequence.sort_values(['person_id', 'visit_rank_order']) \
            .groupby('person_id')['periods'].apply(lambda x: list(itertools.chain(*x))).reset_index()

        _training_data = patient_concept_ids.merge(patient_visit_ids).merge(event_dates).merge(event_periods)
        _training_data = _training_data[_training_data['concept_ids'].apply(len) > 1]
        _training_data.to_pickle(_training_data_path)

    return _training_data


def tokenize_concept_sequences(training_data, tokenizer_path):
    """

    :param training_data:
    :param tokenizer_path:
    :return:
    """
    _tokenizer = ConceptTokenizer()
    _tokenizer.fit_on_concept_sequences(training_data.concept_ids)
    encoded_sequences = _tokenizer.encode(training_data.concept_ids)
    training_data['token_ids'] = encoded_sequences
    pickle.dump(_tokenizer, open(tokenizer_path, 'wb'))

    return _tokenizer, training_data


def train(model_path,
          dataset,
          max_seq_length,
          time_window_size,
          concept_embedding_size,
          vocabulary_size,
          epochs,
          steps_per_epoch,
          learning_rate,
          val_dataset,
          val_steps_per_epoch,
          tf_board_log_path):
    """

    :param model_path:
    :param dataset:
    :param max_seq_length:
    :param time_window_size:
    :param concept_embedding_size:
    :param vocabulary_size:
    :param epochs:
    :param steps_per_epoch:
    :param learning_rate:
    :param val_dataset:
    :param val_steps_per_epoch:
    :param tf_board_log_path:
    :return:
    """
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path, custom_objects=get_custom_objects())
        else:
            optimizer = tf.keras.optimizers.Adam(
                lr=learning_rate, beta_1=0.9, beta_2=0.999)

            model = time_attention_cbow_model(max_seq_length=max_seq_length,
                                              vocabulary_size=vocabulary_size,
                                              concept_embedding_size=concept_embedding_size,
                                              time_window_size=time_window_size, 
                                              time_period_size=2)
            model.compile(
                optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=tf.keras.metrics.sparse_categorical_accuracy)

    model_callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=tf_board_log_path),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            save_best_only=True,
            verbose=1),
        tf.keras.callbacks.LearningRateScheduler(
            CosineLRSchedule(
                lr_high=learning_rate,
                lr_low=1e-8,
                initial_period=10), verbose=1),
    ]

    model.fit(
        dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=model_callbacks,
        validation_data=val_dataset,
        validation_steps=val_steps_per_epoch
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for time-aware concept embedding model')

    parser.add_argument('-i',
                        '--input_folder',
                        dest='input_folder',
                        action='store',
                        help='The path for your input_folder where the raw data is',
                        required=True)

    parser.add_argument('-o',
                        '--output_folder',
                        dest='output_folder',
                        action='store',
                        help='The output folder that stores the domain tables download destination',
                        required=True)

    parser.add_argument('-m',
                        '--max_seq_length',
                        dest='max_seq_length',
                        action='store',
                        type=int,
                        default=100,
                        required=False)

    parser.add_argument('-t',
                        '--time_window_size',
                        dest='time_window_size',
                        action='store',
                        type=int,
                        default=100,
                        required=False)

    parser.add_argument('-c',
                        '--concept_embedding_size',
                        dest='concept_embedding_size',
                        action='store',
                        type=int,
                        default=128,
                        required=False)

    parser.add_argument('-e',
                        '--epochs',
                        dest='epochs',
                        action='store',
                        type=int,
                        default=50,
                        required=False)

    parser.add_argument('-b',
                        '--batch_size',
                        dest='batch_size',
                        action='store',
                        type=int,
                        default=128,
                        required=False)

    parser.add_argument('-lr',
                        '--learning_rate',
                        dest='learning_rate',
                        action='store',
                        type=int,
                        default=2e-4,
                        required=False)

    parser.add_argument('-bl',
                        '--tf_board_log_path',
                        dest='tf_board_log_path',
                        action='store',
                        default='./logs',
                        required=False)

    args = parser.parse_args()

    raw_input_data_path = os.path.join(args.input_folder, 'visit_event_sequence_v2')
    training_data_path = os.path.join(args.output_folder, 'patient_event_sequence.pickle')
    tokenizer_path = os.path.join(args.output_folder, 'tokenizer.pickle')
    model_path = os.path.join(args.output_folder, 'model_time_aware_embeddings.h5')

    training_data = process_raw_input(raw_input_data_path, training_data_path)

    tokenizer, training_data = tokenize_concept_sequences(training_data, tokenizer_path)
    unused_token_id = tokenizer.get_unused_token_id()

    batch_generator = BatchGenerator(patient_event_sequence=training_data,
                                     max_sequence_length=args.max_seq_length,
                                     batch_size=args.batch_size,
                                     unused_token_id=unused_token_id)

    dataset = tf.data.Dataset.from_generator(batch_generator.batch_generator,
                                             output_types=({'target_concepts': tf.int32,
                                                            'target_time_stamps': tf.float32,
                                                            'target_time_periods': tf.float32,
                                                            'context_concepts': tf.int32,
                                                            'context_time_stamps': tf.float32,
                                                            'context_time_periods': tf.float32,
                                                            'mask': tf.int32}, tf.int32))

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE).shuffle(True)

    train(model_path=model_path,
          dataset=dataset,
          max_seq_length=args.max_seq_length,
          time_window_size=args.time_window_size,
          concept_embedding_size=args.concept_embedding_size,
          vocabulary_size=tokenizer.get_vocab_size(),
          epochs=args.epochs,
          steps_per_epoch=batch_generator.get_steps_per_epoch(),
          learning_rate=args.learning_rate,
          val_dataset=dataset,
          val_steps_per_epoch=10,
          tf_board_log_path=args.tf_board_log_path)
