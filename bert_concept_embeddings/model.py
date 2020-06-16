import tensorflow as tf
# +
import tensorflow as tf

from keras_transformer.extras import ReusableEmbedding, TiedOutputEmbedding
from keras_transformer.position import TransformerCoordinateEmbedding
from keras_transformer.transformer import TransformerACT, TransformerBlock

from bert_concept_embeddings.custom_layers import EncoderLayer, TimeSelfAttention, TimeAttention


# -

def time_attention_model(max_seq_length: int,
                         vocabulary_size: int,
                         concept_embedding_size: int):
    """

    :param max_seq_length:
    :param vocabulary_size:
    :param concept_embedding_size:
    :return:
    """
    target_concepts = tf.keras.layers.Input(shape=(1,), dtype='int32', name='target_concepts')

    target_time_stamps = tf.keras.layers.Input(shape=(1,), dtype='int32', name='target_time_stamps')

    context_concepts = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name='context_concepts')

    context_time_stamps = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name='context_time_stamps')

    mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name='mask')

    embedding_layer = tf.keras.layers.Embedding(vocabulary_size, concept_embedding_size, name='embedding_layer',
                                                mask_zero=True)

    time_embedding_layer = TimeAttention(vocab_size=vocabulary_size, target_seq_len=1, context_seq_len=max_seq_length)

    dense_layer = tf.keras.layers.Dense(vocabulary_size)

    softmax_layer = tf.keras.layers.Softmax()

    # shape = (batch_size, seq_len, embedding_size)
    concept_embeddings = embedding_layer(context_concepts)

    # shape = (batch_size, 1, seq_len)
    time_embeddings = time_embedding_layer([target_concepts,
                                            target_time_stamps,
                                            context_time_stamps,
                                            mask])

    # shape = (batch_size, 1, embedding_size)
    combined_embeddings = tf.matmul(time_embeddings, concept_embeddings)

    # shape = (batch_size, 1, vocab_size)
    concept_predictions = softmax_layer(dense_layer(combined_embeddings))

    model = tf.keras.Model(
        inputs=[target_concepts, target_time_stamps, context_concepts, context_time_stamps, mask],
        outputs=[concept_predictions])

    return model


# -
def transformer_bert_model(
        max_seq_length: int,
        vocabulary_size: int,
        concept_embedding_size: int,
        d_model: int,
        num_heads: int,
        transformer_dropout: float = 0.1,
        embedding_dropout: float = 0.6,
        l2_reg_penalty: float = 1e-4):
    """
    Builds a BERT-based model (Bidirectional Encoder Representations
    from Transformers) following paper "BERT: Pre-training of Deep
    Bidirectional Transformers for Language Understanding"
    (https://arxiv.org/abs/1810.04805)

    Depending on the value passed with `use_universal_transformer` argument,
    this function applies either an Adaptive Universal Transformer (2018)
    or a vanilla Transformer (2017) to do the job (the original paper uses
    vanilla Transformer).
    """
    concept_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name='concept_ids')

    time_stamps = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name='time_stamps')

    mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name='mask')

    concept_mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=1)

    l2_regularizer = (tf.keras.regularizers.l2(l2_reg_penalty) if l2_reg_penalty else None)

    embedding_layer = ReusableEmbedding(
        vocabulary_size, concept_embedding_size,
        input_length=max_seq_length,
        name='bpe_embeddings',
        # Regularization is based on paper "A Comparative Study on
        # Regularization Strategies for Embedding-based Neural Networks"
        # https://arxiv.org/pdf/1508.03721.pdf
        embeddings_regularizer=l2_regularizer)

    time_embedding_layer = TimeSelfAttention(vocab_size=vocabulary_size, seq_len=max_seq_length)

    output_layer = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name='concept_prediction_logits')

    output_softmax_layer = tf.keras.layers.Softmax(name='concept_predictions')

    coordinate_embedding_layer = TransformerCoordinateEmbedding(1, name='coordinate_embedding')

    next_step_input, embedding_matrix = embedding_layer(concept_ids)

    # Building a Vanilla Transformer (described in
    # "Attention is all you need", 2017)
    next_step_input = coordinate_embedding_layer(next_step_input, step=0)

    time_attention_logits = time_embedding_layer(concept_ids, time_stamps, mask)
    # pad a dimension to accommodate the head split
    time_attention_logits = tf.expand_dims(time_attention_logits, axis=1)

    for i in range(d_model):
        next_step_input = (
            EncoderLayer(
                name='transformer' + str(i),
                d_model=d_model,
                num_heads=num_heads, dff=2148)
            (next_step_input, concept_mask, time_attention_logits))

        concept_predictions = output_softmax_layer(
            output_layer([next_step_input, embedding_matrix]))

    model = tf.keras.Model(
        inputs=[concept_ids, time_stamps, mask],
        outputs=[concept_predictions])

    return model
