import pickle

import pandas as pd
import os
import math

from keras_transformer.bert import (masked_perplexity,
                                    MaskedPenalizedSparseCategoricalCrossentropy)

from keras.models import load_model
# noinspection PyPep8Naming
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import losses

from bert_concept_embeddings.model import *
from bert_concept_embeddings.utils import CosineLRSchedule

from bert_concept_embeddings.bert_data_generator import ConceptTokenizer, BertBatchGenerator

CONFIDENCE_PENALTY = 0.1
BERT_SPECIAL_TOKENS = ['[MASK]', '[UNUSED]']
MAX_LEN = 512
TIME_WINDOW = 100
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
CONCEPT_EMBEDDING = 128
EPOCH = 100


def compile_new_model():
    optimizer = optimizers.Adam(
        lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999)

    _model = transformer_bert_model(
        max_seq_length=MAX_LEN,
        time_window_size=TIME_WINDOW,
        vocabulary_size=len(tokenizer.tokenizer.index_word) + 1,
        concept_embedding_size=CONCEPT_EMBEDDING,
        depth=5,
        num_heads=8)

    _model.compile(
        optimizer,
        loss=MaskedPenalizedSparseCategoricalCrossentropy(CONFIDENCE_PENALTY),
        metrics={'concept_predictions': masked_perplexity})

    return _model


# +
input_folder = '/data/research_ops/omops/ohdsi_covid/'
output_folder = '/data/research_ops/omops/ohdsi_covid/bert'

training_data_path = os.path.join(input_folder, 'patient_event_sequence.pickle')
tokenizer_output_path = os.path.join(output_folder, 'tokenizer.pickle')
model_output_path = os.path.join(output_folder, 'model_time_aware_embeddings.h5')
# -

training_data = pd.read_pickle(training_data_path)

tokenizer = ConceptTokenizer()
tokenizer.fit_on_concept_sequences(training_data.concept_ids)
encoded_sequences = tokenizer.encode(training_data.concept_ids)
training_data['token_ids'] = encoded_sequences
pickle.dump(tokenizer, open(tokenizer_output_path, 'wb'))

data_generator = BertBatchGenerator(patient_event_sequence=training_data,
                                    mask_token_id=tokenizer.get_mask_token_id(),
                                    unused_token_id=tokenizer.get_unused_token_id(),
                                    max_sequence_length=MAX_LEN,
                                    batch_size=BATCH_SIZE,
                                    first_token_id=tokenizer.get_first_token_index(),
                                    last_token_id=tokenizer.get_last_token_index())

model = compile_new_model()

lr_scheduler = callbacks.LearningRateScheduler(
    CosineLRSchedule(lr_high=LEARNING_RATE, lr_low=1e-8,
                     initial_period=10),
    verbose=1)

model_callbacks = [
    callbacks.ModelCheckpoint(
        filepath=model_output_path,
        save_best_only=True,
        verbose=1),
    lr_scheduler,
]

model.fit_generator(
    generator=data_generator.generate_batches(),
    steps_per_epoch=data_generator.get_steps_per_epoch(),
    epochs=EPOCH,
    callbacks=model_callbacks,
    validation_data=data_generator.generate_batches(),
    validation_steps=10,
)
