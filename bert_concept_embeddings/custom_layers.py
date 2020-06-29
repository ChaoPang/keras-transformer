import tensorflow as tf
import numpy as np

from keras.utils import get_custom_objects


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


def scaled_dot_product_attention(q, k, v, mask, time_attention_logits=None):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (tf.cast(mask, dtype='float32') * -1e9)

    if time_attention_logits is not None:
        scaled_attention_logits += time_attention_logits

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def get_config(self):
        config = super().get_config()
        config['d_model'] = self.d_model
        config['num_heads'] = self.num_heads
        return config

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask, time_attention_logits):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask,
                                                                           time_attention_logits=time_attention_logits)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config['d_model'] = self.d_model
        config['num_heads'] = self.num_heads
        config['dff'] = self.dff
        config['rate'] = self.rate
        return config

    def call(self, x, mask, time_attention_logits):
        attn_output, _ = self.mha(x, x, x, mask, time_attention_logits)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class TimeEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size: int,
                 time_period_size: int,
                 embedding_size: int,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.time_period_size = time_period_size
        self.embedding_size = embedding_size

        self.time_embedding_layer = tf.keras.layers.Embedding(self.vocab_size * self.time_period_size,
                                                              self.embedding_size,
                                                              embeddings_initializer=tf.keras.initializers.zeros,
                                                              name='time_embedding_layer')

    def get_config(self):
        config = super().get_config()
        config['vocab_size'] = self.vocab_size
        config['time_period_size'] = self.time_period_size
        config['embedding_size'] = self.embedding_size
        return config

    def call(self, inputs, **kwargs):
        # Shape = (batch_size, seq_len)
        concept_ids, time_periods = inputs
        # Shape = (batch_size, seq_len)
        time_specific_concept_ids = time_periods * self.vocab_size + concept_ids

        # Shape (batch_size, seq_len, embedding_size)
        time_period_bias = self.time_embedding_layer(time_specific_concept_ids)

        return time_period_bias

    def get_time_period_weights(self):
        return tf.transpose(tf.reshape(self.time_embedding_layer.get_weights()[0],
                                       (self.vocab_size, self.time_period_size, self.embedding_size)), perm=[1, 0, 2])


class TimeAttention(tf.keras.layers.Layer):

    def __init__(self, vocab_size: int,
                 target_seq_len: int,
                 context_seq_len: int,
                 time_window_size: int,
                 return_logits: bool = False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.vocab_size = vocab_size
        self.target_seq_len = target_seq_len
        self.context_seq_len = context_seq_len

        # Save the half window size
        self.half_window_size = int(time_window_size / 2)
        # Pad one for time zero, in which the index event occurred
        self.time_window_size = self.half_window_size * 2 + 1
        self.return_logits = return_logits

        self.embedding_layer = tf.keras.layers.Embedding(self.vocab_size, self.time_window_size,
                                                         embeddings_initializer=tf.keras.initializers.zeros,
                                                         name='time_attention_embedding')
        self.time_attention_bias = self.add_weight(name='time_attention_bias',
                                                   shape=self.time_window_size,
                                                   initializer=tf.keras.initializers.zeros,
                                                   trainable=True)

        self.softmax_layer = tf.keras.layers.Softmax()

    def get_config(self):
        config = super().get_config()
        config['vocab_size'] = self.vocab_size
        config['target_seq_len'] = self.target_seq_len
        config['context_seq_len'] = self.context_seq_len
        config['time_window_size'] = self.time_window_size
        config['return_logits'] = self.return_logits
        return config

    def call(self, inputs, **kwargs):
        """

        :param inputs:
        :param kwargs:
        :return:
        """
        target_concepts, target_time_stamps, context_time_stamps, time_mask = inputs

        concept_time_attentions = self.embedding_layer(target_concepts) + self.time_attention_bias

        return self.apply_time_attentions(concept_time_attentions, context_time_stamps, target_time_stamps, time_mask)

    def apply_time_attentions(self, concept_time_attentions, context_time_stamps, target_time_stamps, time_mask):
        # shape = (batch_size, context_seq_length, target_seq_len)
        multiplied_context_time_stamps = tf.tile(tf.expand_dims(context_time_stamps, axis=-1),
                                                 tf.constant([1, 1, self.target_seq_len]))
        # shape = (batch_size, target_seq_length, context_seq_length)
        time_delta = tf.transpose(multiplied_context_time_stamps - tf.expand_dims(target_time_stamps, axis=1),
                                  perm=[0, 2, 1])
        # Clip the time deltas to fit the time window. E.g. if the time window is 101, the allowed time delta values
        # are between -50 to 50
        time_delta_value_clipped = tf.clip_by_value(time_delta, clip_value_min=-self.half_window_size,
                                                    clip_value_max=self.half_window_size)
        # shape = (batch_size, target_seq_length, context_seq_length, full_time_window_size)
        time_delta_one_hot = tf.one_hot(time_delta_value_clipped + self.half_window_size, self.time_window_size)
        # shape = (batch_size, target_seq_length, time_window_size, 1)
        concept_time_embeddings_expanded = tf.expand_dims(concept_time_attentions, axis=-1)
        # shape = (batch_size, target_seq_length, context_seq_length)
        next_input = tf.squeeze(tf.matmul(time_delta_one_hot, concept_time_embeddings_expanded),
                                axis=-1)
        # add the mask to the scaled tensor.
        if time_mask is not None:
            next_input += (tf.cast(tf.expand_dims(time_mask, axis=1), dtype='float32') * -1e9)
        return next_input if self.return_logits else self.softmax_layer(next_input)


class TimeSelfAttention(TimeAttention):

    def call(self, inputs, **kwargs):
        """

        :param inputs:
        :param kwargs:
        :return:
        """
        batch_concept_sequence = inputs[0]
        batch_time_sequence = inputs[1]
        mask = inputs[2]

        return super().call([batch_concept_sequence, batch_time_sequence, batch_time_sequence, mask])


class TimeSensitiveTimeAttention(TimeAttention):

    def __init__(self, vocab_size: int,
                 target_seq_len: int,
                 target_time_period_size: int,
                 context_seq_len: int,
                 time_window_size: int,
                 return_logits: bool = False, *args, **kwargs):
        super(TimeSensitiveTimeAttention, self).__init__(vocab_size=vocab_size,
                                                         target_seq_len=target_seq_len,
                                                         context_seq_len=context_seq_len,
                                                         time_window_size=time_window_size,
                                                         return_logits=return_logits,
                                                         *args, **kwargs)
        self.target_time_period_size = target_time_period_size
        self.time_embedding_layer = TimeEmbeddingLayer(vocab_size=self.vocab_size,
                                                       time_period_size=self.target_time_period_size,
                                                       embedding_size=self.time_window_size,
                                                       *args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config['target_time_period_size'] = self.target_time_period_size
        return config

    def call(self, inputs, **kwargs):
        """

        :param inputs:
        :param kwargs:
        :return:
        """
        target_concepts, target_time_stamps, target_time_periods, context_time_stamps, time_mask = inputs

        time_attention_modifier = self.time_embedding_layer([target_concepts, target_time_periods])

        concept_time_attentions = self.embedding_layer(
            target_concepts) + self.time_attention_bias + time_attention_modifier

        return self.apply_time_attentions(concept_time_attentions, context_time_stamps, target_time_stamps, time_mask)


get_custom_objects().update({
    'MultiHeadAttention': MultiHeadAttention,
    'EncoderLayer': EncoderLayer,
    'TimeAttention': TimeAttention,
    'TimeSelfAttention': TimeSelfAttention,
    'TimeEmbeddingLayer': TimeEmbeddingLayer,
    'TimeSensitiveTimeAttention': TimeSensitiveTimeAttention
})
