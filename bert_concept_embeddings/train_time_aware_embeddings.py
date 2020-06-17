import os
import itertools
import pickle

import pandas as pd
import numpy as np

from bert_concept_embeddings.bert_data_generator import ConceptTokenizer, BatchGenerator
from bert_concept_embeddings.model import time_attention_model
from bert_concept_embeddings.utils import CosineLRSchedule

import tensorflow as tf

# +
CONFIDENCE_PENALTY = 0.1
BERT_SPECIAL_TOKENS = ['[MASK]', '[UNUSED]']
MAX_LEN = 100
BATCH_SIZE = 128
LEARNING_RATE = 2e-4
CONCEPT_EMBEDDING = 128
EPOCH = 100

INPUT_FOLDER = '/data/research_ops/omops/ohdsi_covid'
# -

if os.path.exists(os.path.join(INPUT_FOLDER, 'patient_event_sequence.pickle')):
    training_data = pd.read_pickle(os.path.join(INPUT_FOLDER, 'patient_event_sequence.pickle'))
else:
    visit_event_sequence_v2 = pd.read_parquet(os.path.join(INPUT_FOLDER, 'visit_event_sequence_v2'))
    visit_event_sequence_v2['concept_id_visit_orders'] = visit_event_sequence_v2['concept_ids'].apply(len) * visit_event_sequence_v2['visit_rank_order'].apply(lambda v: [v])

    patient_concept_ids = visit_event_sequence_v2.sort_values(['person_id', 'visit_rank_order']) \
        .groupby('person_id')['concept_ids'].apply(lambda x: list(itertools.chain(*x))).reset_index()

    patient_visit_ids = visit_event_sequence_v2.sort_values(['person_id', 'visit_rank_order']) \
        .groupby('person_id')['concept_id_visit_orders'].apply(lambda x: list(itertools.chain(*x))).reset_index()

    event_dates = visit_event_sequence_v2.sort_values(['person_id', 'visit_rank_order']) \
        .groupby('person_id')['dates'].apply(lambda x: list(itertools.chain(*x))).reset_index()

    training_data = patient_concept_ids.merge(patient_visit_ids).merge(event_dates)
    training_data = training_data[training_data['concept_ids'].apply(len) > 1]
    training_data.to_pickle(os.path.join(INPUT_FOLDER, 'patient_event_sequence.pickle'))

# +
# all_time_stamps = training_data.dates.explode().astype(int)
# training_data['normalized_dates'] = ((all_time_stamps - all_time_stamps.mean()) / all_time_stamps.std()).groupby(level=0).apply(list)


# +
tokenizer = ConceptTokenizer(special_tokens=BERT_SPECIAL_TOKENS)

tokenizer.fit_on_concept_sequences(training_data.concept_ids)

encoded_sequences = tokenizer.encode(training_data.concept_ids)
mask_token_id, unused_token_id = tokenizer.encode(BERT_SPECIAL_TOKENS)

mask_token_id = mask_token_id[0]
unused_token_id = unused_token_id[0]

training_data['token_ids'] = encoded_sequences

pickle.dump(tokenizer, open('tokenizer.pickle', 'wb'))
# -
batch_generator = BatchGenerator(patient_event_sequence=training_data, 
                           max_sequence_length=MAX_LEN,
                           batch_size=BATCH_SIZE,
                           unused_token_id=unused_token_id)

# +
dataset = tf.data.Dataset.from_generator(batch_generator.batch_generator,
                                         output_types=({'target_concepts': tf.int32, 
                                                        'target_time_stamps': tf.float32, 
                                                        'context_concepts': tf.int32, 
                                                        'context_time_stamps': tf.float32, 
                                                        'mask':tf.int32}, tf.int32))

dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE).shuffle(True).cache()
# -
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    optimizer = tf.keras.optimizers.Adam(
        lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999)

    model = time_attention_model(max_seq_length=MAX_LEN, 
                             vocabulary_size=tokenizer.get_vocab_size() + 1, 
                             concept_embedding_size=CONCEPT_EMBEDDING)
    model.compile(
        optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=tf.keras.metrics.sparse_categorical_accuracy)

model_callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_time_aware_embeddings.h5',
        save_best_only=True, 
        verbose=1),
    tf.keras.callbacks.LearningRateScheduler(
        CosineLRSchedule(
            lr_high=LEARNING_RATE, 
            lr_low=1e-8, 
            initial_period=10), verbose=1),
]

model.summary()

model.fit(
    dataset,
    steps_per_epoch=batch_generator.steps_per_epoch,
    epochs=EPOCH,
    callbacks=model_callbacks,
    validation_data=dataset,
    validation_steps=10
)


