# -*- coding: utf-8 -*-
# BertFC model for text classification

import tensorflow as tf
from kyber.modules import BertEncoder
from tensorflow.python.keras.layers import Input, Dense

class BertFC(tf.keras.Model):
    def __init__(self, num_classes, model_path, bert_config=None):
        """
        :param num_classes: number of categories
        :param model_path: path of tf bert pre-trained model checkpoint, ending with xxx.ckpt
        :param config_json: BertConfig instance to initialize bert encoder
        """
        super(BertFC, self).__init__()
        self.num_classes = num_classes
        self.bert_encoder = BertEncoder(bert_config)
        self.bert_encoder.load_weights_from_checkpoint(model_path)
        print("Bert pre-trained models loading done")
        self.linear_out = Dense(units=self.num_classes, activation='softmax')

        # init with Model API to build
        inputs_ids = Input(shape=(None,), dtype=tf.float32)
        type_ids = Input(shape=(None,), dtype=tf.float32)
        inputs = [inputs_ids, type_ids]
        super(BertFC, self).__init__(inputs=inputs, outputs=self.call(inputs))

    def call(self, inputs, **kwargs):
        hidden_states, pool_output = self.bert_encoder(inputs)
        outputs = self.linear_out(pool_output)
        return outputs


