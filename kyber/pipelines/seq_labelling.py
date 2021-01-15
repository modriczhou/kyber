#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
from modules.pipeline import Pipeline
from data_utils import Field
from models import BertNer
from config import *

class BertTokenClfPipeline(Pipeline):
    # Pipeline example for bert for token classification
    """
        Task:   sequence labelling
        Model:  bert + fc
    """
    def build_field(self, **kwargs):
        news_txt_field = Field(name='text', tokenizer=kwargs['tokenizer'], seq_flag=True)

        label_filed = Field(name='label', tokenizer=None, seq_flag=True, is_target=True)

        self.fields_dict = {"text": news_txt_field, "label": label_filed} ## 顺序和column一致
        self.vocab_group = [["text"],["label"]]

    def build_model(self, ):
        self.model = BertNer(num_labels=self.num_classes,
                            model_path=self.bert_pretrained_path['model_path'],
                            bert_config=BertConfig(self.bert_pretrained_path['dict_path']))
        self.model.summary()

    def train(self, epochs, callbacks):
        """
         Build model, loss, optimizer and train
        :param epochs: number of epochs in training
        :param callbacks:
        :return:
        """
        # opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        # self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        print("start fit")
        opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        self.model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.train_iter.forfit(), steps_per_epoch=len(self.train_iter), epochs=epochs, \
                            validation_data=self.dev_iter.forfit(), validation_steps=len(self.dev_iter),
                            callbacks=callbacks)
