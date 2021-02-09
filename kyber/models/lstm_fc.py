#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from kyber.modules.encoders import RnnEncoder
from tensorflow.python.keras.layers import Input, Dense, Embedding, LSTM

class RnnFC(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_classes, num_units, **kwargs):
        super(LstmFC, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)
        self.lstm_encoder = RnnEncoder(cell_type="lstm",units=num_units, **kwargs)
        self.linear_out = Dense(units=self.num_classes, activation='softmax')

    def call(self, inputs, *args):
        embedding = self.embedding(inputs)
        encoding = self.lstm_encoder(embedding)
        output = self.linear_out(encoding)
        return output

    def summary(self, *args):
        inputs = Input(shape=(self.seq_len, ), dtype=tf.float32)
        tf.keras.Model(inputs=inputs, outputs=self.call(inputs)).summary()
