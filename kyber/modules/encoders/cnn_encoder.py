#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Embedding, Conv2D, MaxPool2D
from tensorflow.keras.layers import Reshape, Flatten, Concatenate, Dropout

class CnnEncoder(layers.Layer):
    """
    Basic Text CNN encoders Implementation
    """
    def __init__(self, embedding_dim, num_filters=512, input_length=256, drop_ratio=0.5, filter_sizes=None):
        super(CnnEncoder, self).__init__()
        # self.embedding_exclusive = embedding_exclusive
        # if not self.embedding_exclusive:
        #     self.embedding = Embedding(input_dim=vocab_size, output_dim=)
        self.num_filters = num_filters
        self.input_length = input_length
        self.filter_sizes = filter_sizes if filter_sizes else [3,4,5]
        self.drop_ratio = drop_ratio
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self._reshape = Reshape((self.input_length, self.embedding_dim, 1)) # shape:(None, seq_len, self.embedding, 1)
        self._conv_layer_0 = Conv2D(filters=self.num_filters, kernel_size=(self.filter_sizes[0], self.embedding_dim), \
                             padding='valid', kernel_initializer='normal', activation='relu')
        self._conv_layer_1 = Conv2D(filters=self.num_filters, kernel_size=(self.filter_sizes[1], self.embedding_dim), \
                             padding='valid', kernel_initializer='normal', activation='relu')
        self._conv_layer_2 = Conv2D(filters=self.num_filters, kernel_size=(self.filter_sizes[2], self.embedding_dim), \
                             padding='valid', kernel_initializer='normal', activation='relu')
        # self.reshape = Reshape(())
        self._maxpool_layer_0 = MaxPool2D(pool_size=(self.input_length + 1 - self.filter_sizes[0], 1), strides=(1,1), padding='valid')
        self._maxpool_layer_1 = MaxPool2D(pool_size=(self.input_length + 1 - self.filter_sizes[1], 1), strides=(1, 1), padding='valid')
        self._maxpool_layer_2 = MaxPool2D(pool_size=(self.input_length + 1 - self.filter_sizes[2], 1), strides=(1, 1), padding='valid')

        super(CnnEncoder, self).build(input_shape)

    def call(self, embeds, **kwargs):
        """
        :param embeds: batch embeds result
        :return:
        """
        reshaped = self._reshape(embeds)
        conv_0 = self._conv_layer_0(reshaped)
        conv_1 = self._conv_layer_1(reshaped)
        conv_2 = self._conv_layer_2(reshaped)

        maxpool_0 = self._maxpool_layer_0(conv_0)
        maxpool_1 = self._maxpool_layer_1(conv_1)
        maxpool_2 = self._maxpool_layer_2(conv_2)
        # concatenated = maxpool_1
        concatenated = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten()(concatenated)
        dropout = Dropout(self.drop_ratio)(flatten)

        return dropout

    # def __call__(self, *args, **kwargs):