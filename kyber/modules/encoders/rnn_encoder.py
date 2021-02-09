# -*- coding: utf-8 -*-
"""
实现rnn_encoder，可定义单向/双向，选择LSTM, GRU或普通的rnn
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, Bidirectional, GRU, SimpleRNN

class RnnEncoder(layers.Layer):
    def __init__(self, cell_type, num_units, bi_directional, backward_layer=None, **kwargs):
        super(RnnEncoder, self).__init__()
        self.cell_type = cell_type
        self.num_units = num_units
        self.bi_directional = bi_directional
        self.kwargs = kwargs
        self.backward_layer = backward_layer

    def build(self, input_shape):
        if self.cell_type == "lstm":
            self.encoder = LSTM(self.num_units)
        elif self.cell_type == "gru":
            self.encoder = GRU(self.num_units)
        else:
            self.encoder = SimpleRNN(self.num_units)

        if self.bi_directional:
            self.encoder = Bidirectional(self.encoder)
        super(RnnEncoder, self).build(input_shape)

    def call(self, embeds, **kwargs):
        return self.encoder(embeds)