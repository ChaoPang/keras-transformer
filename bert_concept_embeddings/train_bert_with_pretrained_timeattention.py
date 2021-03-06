import pickle

import pandas as pd
import os

from keras_transformer.bert import (masked_perplexity,
                                    MaskedPenalizedSparseCategoricalCrossentropy)

from tensorflow.keras import optimizers
from tensorflow.keras import callbacks

from bert_concept_embeddings.model import *
from bert_concept_embeddings.utils import CosineLRSchedule
from bert_concept_embeddings.custom_layers import get_custom_objects

from bert_concept_embeddings.bert_data_generator import ConceptTokenizer, BertBatchGenerator

CONFIDENCE_PENALTY = 0.1
BERT_SPECIAL_TOKENS = ['[MASK]', '[UNUSED]']
MAX_LEN = 100
TIME_WINDOW = 100
BATCH_SIZE = 256
LEARNING_RATE = 2e-4
CONCEPT_EMBEDDING = 128
EPOCH = 10


def compile_new_model():
    optimizer = optimizers.Adam(
        lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999)

    _model = transformer_bert_model(
        max_seq_length=MAX_LEN,
        time_window_size=TIME_WINDOW,
        vocabulary_size=len(tokenizer.tokenizer.index_word) + 1,
        concept_embedding_size=CONCEPT_EMBEDDING,
        depth=5,
        num_heads=8,
        time_attention_trainable=False)

    _model.compile(
        optimizer,
        loss=MaskedPenalizedSparseCategoricalCrossentropy(CONFIDENCE_PENALTY),
        metrics={'concept_predictions': masked_perplexity})

    return _model


# +
data_folder = '/data/research_ops/omops/omop_2020q1/output'
time_attention_attention_folder = '/data/research_ops/omops/omop_2020q1/output'
bert_model_folder = '/data/research_ops/omops/omop_2020q1/bert'

training_data_path = os.path.join(data_folder, 'patient_event_sequence.pickle')
tokenizer_path = os.path.join(data_folder, 'tokenizer.pickle')
bert_model_path = os.path.join(bert_model_folder, 'model_time_aware_embeddings.h5')
time_attention_model_path = os.path.join(time_attention_attention_folder, 'model_time_aware_embeddings.h5')
# -

training_data = pd.read_pickle(training_data_path)

tokenizer = ConceptTokenizer()
tokenizer.fit_on_concept_sequences(training_data.concept_ids)
encoded_sequences = tokenizer.encode(training_data.concept_ids)
training_data['token_ids'] = encoded_sequences
pickle.dump(tokenizer, open(tokenizer_path, 'wb'))

data_generator = BertBatchGenerator(patient_event_sequence=training_data,
                                    mask_token_id=tokenizer.get_mask_token_id(),
                                    unused_token_id=tokenizer.get_unused_token_id(),
                                    max_sequence_length=MAX_LEN,
                                    batch_size=BATCH_SIZE,
                                    first_token_id=tokenizer.get_first_token_index(),
                                    last_token_id=tokenizer.get_last_token_index())

dataset = tf.data.Dataset.from_generator(data_generator.batch_generator,
                                         output_types=({'masked_concept_ids': tf.int32,
                                                        'concept_ids': tf.int32,
                                                        'time_stamps': tf.int32,
                                                        'mask': tf.int32}, tf.int32))

another_strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
with another_strategy.scope():
    time_attention_model = tf.keras.models.load_model(time_attention_model_path,
                                                      custom_objects=dict(**get_custom_objects()))
    pretrained_embedding_layer = time_attention_model.get_layer('time_attention').embedding_layer
    weights = pretrained_embedding_layer.get_weights()

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
    if os.path.exists(bert_model_path):
        model = tf.keras.models.load_model(bert_model_path, custom_objects=get_custom_objects())
    else:
        model = compile_new_model()
        self_attention_layer_name = [layer.name for layer in model.layers if 'time_self_attention' in layer.name]
        if self_attention_layer_name:
            model.get_layer(self_attention_layer_name[0]).set_weights(weights)

lr_scheduler = callbacks.LearningRateScheduler(
    CosineLRSchedule(lr_high=LEARNING_RATE, lr_low=1e-8,
                     initial_period=10),
    verbose=1)

model_callbacks = [
    callbacks.ModelCheckpoint(
        filepath=bert_model_path,
        save_best_only=True,
        verbose=1),
    lr_scheduler,
]

model.fit(
    dataset,
    steps_per_epoch=data_generator.get_steps_per_epoch() + 1,
    epochs=EPOCH,
    callbacks=model_callbacks,
    validation_data=dataset.shard(10, 1),
    validation_steps=10,
)
