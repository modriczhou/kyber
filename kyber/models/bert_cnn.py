#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from kyber.modules.encoders import CnnEncoder
from tensorflow.keras.layers import Input, Dense, Embedding

class BertCNN(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_classes, num_filters=512, seq_len=512):
        super(BertCNN, self).__init__()
        # print(vocab_size, embedding_dim, num_classes)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)

        self.cnn_encoder = CnnEncoder(embedding_dim=self.embedding_dim, num_filters=self.num_filters, input_length=self.seq_len)
        self.linear_out = Dense(units=self.num_classes, activation='softmax')

    def call(self, inputs, **kwargs):
        # print(inputs)
        # print(self.seq_len)
        # print(inputs)
        #embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)(inputs)
        embedding = self.embedding(inputs)
        #print("emvedding shape:", embedding.shape)
        # print(embedding)
        encoding = self.cnn_encoder(embedding)
        output = self.linear_out(encoding)

        return output

    def summary(self):
        inputs = Input(shape=(self.seq_len, ), dtype=tf.float32)
        tf.keras.Model(inputs = inputs, outputs = self.call(inputs)).summary()