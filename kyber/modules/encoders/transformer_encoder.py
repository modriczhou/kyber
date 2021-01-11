#h!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Yuansheng Zhou

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from layers.transformer_layers import *

"""
## Implement a Transformer block as a layer
"""

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.dropout1 = layers.Dropout(rate)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)

        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.dropout2 = layers.Dropout(rate)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)


    def call(self, inputs, mask=None, **kwargs):
        attn_output = self.att(inputs, mask=mask)
        # print("attn_output size:", attn_output.shape)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        # print("ff_output:",ffn_output.shape)
        return self.layernorm2(out1 + ffn_output)


