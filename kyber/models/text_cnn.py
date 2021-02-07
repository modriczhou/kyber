# -*- coding: utf-8 -*-
"""
定义TextCNN Model
CNN Encoder为Kim—CNN经典结构
Embedding->TextCNN->Dense
"""
import tensorflow as tf
from kyber.modules.encoders import CnnEncoder
from tensorflow.python.keras.layers import Input, Dense, Embedding

class TextCNN(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_classes, num_filters=512, seq_len=512, filter_sizes=None):
        super(TextCNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.filter_sizes = filter_sizes
        self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)
        self.cnn_encoder = CnnEncoder(embedding_dim=self.embedding_dim,
                                      num_filters=self.num_filters,
                                      input_length=self.seq_len,
                                      filter_sizes = self.filter_sizes)
        self.linear_out = Dense(units=self.num_classes, activation='softmax')

    def call(self, inputs, *args):
        embedding = self.embedding(inputs)
        encoding = self.cnn_encoder(embedding)
        output = self.linear_out(encoding)
        return output

    def summary(self, *args):
        inputs = Input(shape=(self.seq_len, ), dtype=tf.float32)
        tf.keras.Model(inputs=inputs, outputs=self.call(inputs)).summary()