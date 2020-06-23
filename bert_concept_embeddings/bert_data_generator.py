# +
import random
from itertools import islice, chain
from typing import List, Callable, Optional, Sequence

import numpy as np
# noinspection PyPep8Naming
from keras import backend as K
from keras.utils import get_custom_objects
# -

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

BERT_SPECIAL_TOKENS = ['[MASK]', '[UNUSED]']


class ConceptTokenizer:
    unused_token = ['[UNUSED]']
    mask_token = ['[MASK]']

    def __init__(self, special_tokens: Optional[Sequence[str]] = None, oov_token='0'):
        self.special_tokens = special_tokens
        self.tokenizer = Tokenizer(oov_token=oov_token, filters='', lower=False)

    def fit_on_concept_sequences(self, concept_sequences):
        self.tokenizer.fit_on_texts(concept_sequences)
        self.tokenizer.fit_on_texts(self.mask_token)
        self.tokenizer.fit_on_texts(self.unused_token)
        if self.special_tokens is not None:
            self.tokenizer.fit_on_texts(self.special_tokens)

    def encode(self, concept_sequences):
        return self.tokenizer.texts_to_sequences(concept_sequences)

    def decode(self, concept_sequence_token_ids):
        return self.tokenizer.sequences_to_texts(concept_sequence_token_ids)

    def get_all_token_indexes(self):
        all_keys = set(self.tokenizer.index_word.keys())

        if self.tokenizer.oov_token is not None:
            all_keys.remove(self.tokenizer.word_index[self.tokenizer.oov_token])

        if self.special_tokens is not None:
            excluded = set([self.tokenizer.word_index[special_token] for special_token in self.special_tokens])
            all_keys = all_keys - excluded
        return all_keys

    def get_first_token_index(self):
        return min(self.get_all_token_indexes())

    def get_last_token_index(self):
        return max(self.get_all_token_indexes())

    def get_vocab_size(self):
        # + 1 because oov_token takes the index 0
        return len(self.tokenizer.index_word) + 1

    def get_unused_token_id(self):
        unused_token_id = self.encode(self.unused_token)
        while isinstance(unused_token_id, list):
            unused_token_id = unused_token_id[0]
        return unused_token_id

    def get_mask_token_id(self):
        mask_token_id = self.encode(self.mask_token)
        while isinstance(mask_token_id, list):
            mask_token_id = mask_token_id[0]
        return mask_token_id


class BatchGenerator:

    def __init__(self, patient_event_sequence,
                 unused_token_id: int,
                 max_sequence_length: int,
                 batch_size: int):

        self.patient_event_sequence = patient_event_sequence
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.unused_token_id = unused_token_id

    def batch_generator(self):
        training_example_generator = self.data_generator()
        while True:
            next_bunch_of_examples = islice(training_example_generator, self.batch_size)
            target_concepts, target_time_stamps, context_concepts, context_time_stamps, labels = zip(
                *list(next_bunch_of_examples))

            target_concepts = np.asarray(target_concepts)
            target_time_stamps = np.asarray(target_time_stamps)
            context_concepts = pad_sequences(context_concepts, maxlen=self.max_sequence_length, padding='post',
                                             value=self.unused_token_id)
            context_time_stamps = pad_sequences(context_time_stamps, maxlen=self.max_sequence_length, padding='post',
                                                value=0, dtype='float32')
            mask = (context_concepts == self.unused_token_id).astype(int)

            yield ({'target_concepts': target_concepts,
                    'target_time_stamps': target_time_stamps,
                    'context_concepts': context_concepts,
                    'context_time_stamps': context_time_stamps,
                    'mask': mask}, labels)

    def data_generator(self):
        half_window_size = int(self.max_sequence_length / 2)
        while True:
            for tup in self.patient_event_sequence.itertuples():
                concept_ids = tup.token_ids
                dates = tup.dates
                for i, concept_id in enumerate(concept_ids):
                    left_index = i - half_window_size if i - half_window_size > 0 else 0
                    right_index = i + 1 + half_window_size
                    target_concepts = [concept_id]
                    target_time_stamps = [dates[i]]
                    context_concepts = concept_ids[left_index: i] + concept_ids[i + 1: right_index]
                    context_time_stamps = dates[left_index: i] + dates[i + 1: right_index]

                    yield (target_concepts, target_time_stamps, context_concepts, context_time_stamps, target_concepts)

    def get_steps_per_epoch(self):
        return self.estimate_data_size() // self.batch_size

    def estimate_data_size(self):
        return len(self.patient_event_sequence.token_ids.explode())


class NegativeSamplingBatchGenerator(BatchGenerator):

    def __init__(self,
                 num_of_negative_samples: int,
                 negative_sample_factor: float,
                 first_token_id: int,
                 last_token_id: int,
                 *args, **kwargs):
        super(NegativeSamplingBatchGenerator, self).__init__(*args, **kwargs)
        self.num_of_negative_samples = num_of_negative_samples
        self.first_token_id = first_token_id
        self.last_token_id = last_token_id

        # build the token negative sampling probability distribution
        all_tokens = self.patient_event_sequence.token_ids.explode()
        self.token_prob_dist = np.power(all_tokens.value_counts(), negative_sample_factor)
        self.token_prob_dist = self.token_prob_dist / np.sum(self.token_prob_dist)

    def data_generator(self):
        training_example_generator = super().data_generator()
        for positive_example in training_example_generator:
            negative_example_generator = self.negative_sample_generator(positive_example)
            for next_example in negative_example_generator:
                yield next_example

    def negative_sample_generator(self, next_example):

        target_concepts, target_time_stamps, context_concepts, context_time_stamps, labels = next_example
        # Yield the positive example
        yield (target_concepts, target_time_stamps, context_concepts, context_time_stamps, [1])

        all_token_ids = list(range(self.first_token_id, self.last_token_id + 1))
        samples = set()
        while len(samples) < self.num_of_negative_samples:
            candidates = np.random.choice(all_token_ids, self.num_of_negative_samples, False, self.token_prob_dist)
            samples.update(np.setdiff1d(candidates, target_concepts + context_concepts))

        # yield the negative examples
        for negative_sample in samples:
            yield ([negative_sample], target_time_stamps, context_concepts, context_time_stamps, [0])

    def estimate_data_size(self):
        return super().estimate_data_size() * (1 + self.num_of_negative_samples)


class BertBatchGenerator(BatchGenerator):
    """
    This class generates batches for a BERT-based language model
    in an abstract way, by using an external function sampling
    sequences of token IDs of a given length.
    """

    def __init__(self,
                 mask_token_id: int,
                 first_token_id: int,
                 last_token_id: int,
                 *args, **kwargs):
        super(BertBatchGenerator, self).__init__(*args, **kwargs)
        self.mask_token_id = mask_token_id
        self.first_token_id = first_token_id
        self.last_token_id = last_token_id

    def generate_batches(self):
        training_example_generator = self.data_generator()
        while True:
            next_bunch_of_examples = islice(training_example_generator, self.batch_size)
            output_mask, sequence, masked_sequence, time_stamp_sequence = zip(*list(next_bunch_of_examples))

            sequence = pad_sequences(np.asarray(sequence), maxlen=self.max_sequence_length, padding='post',
                                     value=self.unused_token_id)
            masked_sequence = pad_sequences(np.asarray(masked_sequence), maxlen=self.max_sequence_length,
                                            padding='post',
                                            value=self.unused_token_id)
            time_stamp_sequence = pad_sequences(np.asarray(time_stamp_sequence), maxlen=self.max_sequence_length,
                                                padding='post',
                                                value=0, dtype='float32')
            mask = (sequence == self.unused_token_id).astype(int)
            combined_label = np.stack([sequence, output_mask], axis=-1)

            yield ({'concept_ids': sequence,
                    'masked_concept_ids': masked_sequence,
                    'time_stamps': time_stamp_sequence,
                    'mask': mask}, [combined_label])

    def data_generator(self):
        while True:
            for tup in self.patient_event_sequence.itertuples():
                concept_ids = tup.token_ids
                dates = tup.dates
                for i, concept_id in enumerate(concept_ids):
                    right_index = i + self.max_sequence_length

                    sequence = concept_ids[i: right_index]
                    time_stamp_sequence = dates[i: right_index]

                    masked_sequence = sequence.copy()
                    output_mask = np.zeros((self.max_sequence_length,), dtype=int)

                    for word_pos in range(0, len(sequence)):
                        if sequence[word_pos] == self.unused_token_id:
                            break
                        if random.random() < 0.15:
                            dice = random.random()
                            if dice < 0.8:
                                masked_sequence[word_pos] = self.mask_token_id
                            elif dice < 0.9:
                                masked_sequence[word_pos] = random.randint(
                                    self.first_token_id, self.last_token_id)
                            # else: 10% of the time we just leave the word as is
                            output_mask[word_pos] = 1

                    yield (output_mask, sequence, masked_sequence, time_stamp_sequence)
