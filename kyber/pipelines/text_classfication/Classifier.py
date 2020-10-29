#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/11 10:24
# @Author  : Yuansheng Zhou
# @Site    : 
# @File    : Classifier.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.keras.layers import Embedding

class Classifier(tf.keras.Model):
    def __init__(self, fields_dict, input_fields, output_field, vocab_group, embed_dims):
        """

        :param fields_dict: dict of fields for this data {field_name: filed_class}
        :input_fields:¬
        :param output_field:
        :param embed_dims: dict of embed dims {field_name: embed_dim}
        """
        super(Classifier, self).__init__()

        self.embeddings_list = []

        for input_field in input_fields:
            embedding = Embedding(input_dim=len(input_field._vocab), output_dim=embed_dims[input_field.name])
            self.embeddings_list.append(embedding)


    def call(self, inputs):
        """

        :param inputs: input batch,
        :return:
        """












