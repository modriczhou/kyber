#h!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Yuansheng Zhou
import tensorflow as tf
from tensorflow.keras.layers import Layer


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

"""
## Implement multi head self attention as a Keras layer
"""

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value,mask=None):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        # key_masks = tf.tile(mask, [self.num_heads, 1])
        # key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

        # outputs = tf.where(tf.equal(mask, 0), paddings, outputs)
        # if mask is not None:

        # print("mask_size:",mask.shape)

        attn_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.float32)
        # attn_mask = K.repeat(mask, dim_key)
        # print("attn_mask:",attn_mask.shape)

        # attn_mask = tf.transpose(attn_mask, [0,1,3,2])
        # print("attn_mask after trans:", attn_mask.shape)
        attn_mask = (1.0 - attn_mask) * -10000000.0

        scaled_score = scaled_score + attn_mask

        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        # print("query size：",query.shape)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)

        # mask= K.repeat(mask, self.num_heads)

        attention, weights = self.attention(query, key, value, mask)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


"""
## Implement a Transformer block as a layer
"""

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim,rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training, mask=None):
        attn_output = self.att(inputs, mask)
        # print("attn_output size:", attn_output.shape)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # print("ff_output:",ffn_output.shape)
        return self.layernorm2(out1 + ffn_output)


"""
## Implement embedding layer
Two seperate embedding layers, one for tokens, one for token index (positions).
"""

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, emded_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=emded_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=emded_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class MeanPool(layers.Layer):
  def __init__(self, **kwargs):
      self.supports_masking = True
      super(MeanPool, self).__init__(**kwargs)

  def compute_mask(self, input, input_mask=None):
      # do not pass the mask to the next layers
      return None

  def call(self, x, mask=None):
      if mask is not None:
          # mask (batch, time)
          mask = K.cast(mask, K.floatx())
          # mask (batch, x_dim, time)
          mask = K.repeat(mask, x.shape[-1])
          # mask (batch, time, x_dim)
          mask = tf.transpose(mask, [0,2,1])
          x = x * mask
          # print(mask)
      return K.sum(x, axis=1) / K.sum(mask, axis=1)

  def compute_output_shape(self, input_shape):
      # remove temporal dimension
      return (input_shape[0], input_shape[2])