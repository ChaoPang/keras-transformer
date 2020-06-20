import pandas as pd
import os
import math

from keras_transformer.bert import (masked_perplexity,
                                    MaskedPenalizedSparseCategoricalCrossentropy)

from keras.models import load_model
# noinspection PyPep8Naming
from keras import optimizers
from keras import callbacks
from keras import losses

from bert_concept_embeddings.model import *
from bert_concept_embeddings.utils import CosineLRSchedule

from bert_concept_embeddings.bert_data_generator import ConceptTokenizer, BertBatchGenerator
from keras.preprocessing.sequence import pad_sequences

CONFIDENCE_PENALTY = 0.1
BERT_SPECIAL_TOKENS = ['[MASK]', '[UNUSED]']
MAX_LEN = 512
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
CONCEPT_EMBEDDING = 128
EPOCH = 100


def compile_new_model():
    optimizer = optimizers.Adam(
        lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999)

    _model = transformer_bert_model(
        max_seq_length=MAX_LEN,
        vocabulary_size=len(tokenizer.tokenizer.index_word) + 1,
        concept_embedding_size=CONCEPT_EMBEDDING,
        d_model=5,
        num_heads=8)

    _model.compile(
        optimizer,
        loss=MaskedPenalizedSparseCategoricalCrossentropy(CONFIDENCE_PENALTY),
        metrics={'concept_predictions': masked_perplexity})

    return _model


input_folder = '/data/research_ops/omops/ohdsi_covid/'
visit_event_sequence = pd.read_parquet(os.path.join(input_folder, 'visit_event_sequence'))
visit_event_sequence = visit_event_sequence[
    (visit_event_sequence['collection_size'] >= 10) & (visit_event_sequence['collection_size'] <= 100)]
visit_sequence = visit_event_sequence['concept_list'].apply(lambda seq: list(set(seq.split(' '))))

# +
tokenizer = ConceptTokenizer(special_tokens=BERT_SPECIAL_TOKENS)

tokenizer.fit_on_concept_sequences(visit_sequence)

encoded_visit_sequence = tokenizer.encode(visit_sequence)
mask_token_id, unused_token_id = tokenizer.encode(BERT_SPECIAL_TOKENS)

mask_token_id = mask_token_id[0]
unused_token_id = unused_token_id[0]

first_normal_token_id = tokenizer.get_first_token_index()
last_normal_token_id = tokenizer.get_last_token_index()
# -

padded_visit_sequences = pad_sequences(encoded_visit_sequence, maxlen=MAX_LEN, padding='post', value=unused_token_id)

data_generator = BertBatchGenerator(padded_visit_sequences,
                                    mask_token_id=mask_token_id,
                                    unused_token_id=unused_token_id,
                                    max_sequence_length=MAX_LEN,
                                    batch_size=BATCH_SIZE,
                                    first_normal_token_id=first_normal_token_id,
                                    last_normal_token_id=last_normal_token_id)

model = compile_new_model()

lr_scheduler = callbacks.LearningRateScheduler(
    CosineLRSchedule(lr_high=LEARNING_RATE, lr_low=1e-8,
                     initial_period=10),
    verbose=1)

model_callbacks = [
    callbacks.ModelCheckpoint(
        filepath='model_path.h5',
        save_best_only=True,
        verbose=1),
    lr_scheduler,
]

model.fit_generator(
    generator=data_generator.generate_batches(),
    steps_per_epoch=data_generator.steps_per_epoch,
    epochs=EPOCH,
    callbacks=model_callbacks,
    validation_data=data_generator.generate_batches(),
    validation_steps=10,
)
