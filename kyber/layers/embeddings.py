#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/10 11:11
# @Author  : Yuansheng Zhou
# @Site    : 
# @File    : embeddings.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras import layers
from kyber.utils import shape_list

class BertEmbeddings(layers.Layer):
    """
    Bert Embedding consists three parts:
    1. word embedding
    2. position embeddings
    3. token type (segment) embeddings
    """
    def __init__(self, config, **kwargs):
        super(BertEmbeddings, self).__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range
        self.layer_norm_eps = config.layer_norm_eps
        self.initializer = tf.keras.initializers.TruncatedNormal(config.initializer_range)
        self.max_position_embeddings = config.max_position_embeddings # max length of sequence, default: 512
        self.type_vocab_size = config.type_vocab_size # token type size default:2

    def build(self, input_shape):
        self.word_embeddings = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.hidden_size,
            embeddings_initializer=self.initializer,
            name="word_embeddings"
        )
        self.token_type_embeddings = Embedding(
            input_dim=self.type_vocab_size,
            output_dim=self.hidden_size,
            embeddings_initializer=self.initializer,
            name="token_type_embeddings"
        )
        self.position_embeddings = Embedding(
            input_dim=self.max_position_embeddings,
            output_dim=self.hidden_size,
            embeddings_initializer=self.initializer,
            name="position_embeddings"
        )

        # self.layer_norm = layers.LayerNormalization(self.layer_norm_eps)

        super(BertEmbeddings, self).build(input_shape)



    def call(self, inputs, **kwargs):
        input_ids,token_type_ids = inputs[0], inputs[1]
        shape = shape_list(input_ids)
        seq_len = shape[1]
        position_ids = tf.range(start=0, limit=seq_len, delta=1)
        word_embeds = self.word_embeddings(input_ids)
        pos_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        return word_embeds + pos_embeds + token_type_embeds

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, emded_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=emded_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=emded_dim)

    def call(self, x, **kwargs):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
