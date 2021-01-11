# -*- coding: utf-8 -*-

import tensorflow as tf
import transformers
from modules import BertEncoder
from tensorflow.python.keras.layers import Input, Dense
from utils.bert_tokenizer import Tokenizer

class BertFC(tf.keras.Model):
    def __init__(self, num_classes, model_path, config_json=None):
        super(BertFC, self).__init__()
        self.num_classes = num_classes
        self.bert_encoder = BertEncoder(config_json)
        self.bert_encoder.load_weights_from_checkpoint(model_path)
        print("load_done")
        self.linear_out = Dense(units=self.num_classes, activation='softmax')
        # self.tokenizer = Tokenizer(vocab_path, do_lower_case=True)

        inputs_ids = Input(shape=(None,), dtype=tf.float32)
        type_ids = Input(shape=(None,), dtype=tf.float32)
        inputs = [inputs_ids, type_ids]

        super(BertFC, self).__init__(inputs=inputs, outputs=self.call(inputs))

    def call(self, inputs, **kwargs):
        hidden_states, pool_output = self.bert_encoder(inputs)
        #print("bert_embeddings：", hidden_states)
        #print("pool_output",pool_output)
        outputs = self.linear_out(pool_output)
        #print(outputs)
        return outputs


if __name__ == '__main__':
    model_path = "/Users/James/Study/pretrained_models/bert/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_model.ckpt"
    dict_path = "/Users/James/Study/pretrained_models/bert/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_config.json"
