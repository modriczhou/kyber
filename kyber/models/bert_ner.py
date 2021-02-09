#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from kyber.modules import BertEncoder
from tensorflow.python.keras.layers import Input, Dense, Dropout

class BertNer(tf.keras.Model):

    def __init__(self, num_labels, model_path, bert_config=None):
        """
        :param num_classes: number of categories
        :param model_path: path of tf bert pre-trained model checkpoint, ending with xxx.ckpt
        :param config_json: BertConfig instance to initialize bert encoder
        """
        super(BertNer, self).__init__()
        self.num_labels = num_labels
        self.bert_encoder = BertEncoder(bert_config, add_pooling_layer=False)
        self.bert_encoder.load_weights_from_checkpoint(model_path)
        print("Bert pre-trained models loading done")
        self.dropout = Dropout(bert_config.hidden_dropout_prob)
        self.linear_out = Dense(units=self.num_labels, activation='softmax')

        # init with Model API to build
        inputs_ids = Input(shape=(None,), dtype=tf.float32)
        type_ids = Input(shape=(None,), dtype=tf.float32)
        inputs = [inputs_ids, type_ids]
        super(BertNer, self).__init__(inputs=inputs, outputs=self.call(inputs))

    def call(self, inputs, **kwargs):
        sequence_output = self.bert_encoder(inputs)
        sequence_output = self.dropout(sequence_output)

        outputs = self.linear_out(sequence_output)
        # outputs = TimeDistributed(Dense(self.num_labels, activation='softmax'))(sequence_output)
        # TimeDisrtibuted 没有起到降低参数的作用，和直接用dense似乎效果一样

        return outputs