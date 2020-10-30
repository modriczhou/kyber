# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding

class FastText(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_classes, num_filters=512, seq_len=512):
        super(FastText, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)

        self.linear_out = Dense(units=self.num_classes, activation='softmax')

    def call(self, inputs, *args):
        embedding = self.embedding(inputs)

        encoding = self.cnn_encoder(embedding)
        output = self.linear_out(encoding)

        return output

    def summary(self, *args):
        inputs = Input(shape=(self.seq_len, ), dtype=tf.float32)
        tf.keras.Model(inputs=inputs, outputs=self.call(inputs)).summary()