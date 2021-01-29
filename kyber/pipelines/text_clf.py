#!/usr/bin/env python
# -*- coding: utf-8 -*-
from models import *
import tensorflow as tf
from config import *
from data_utils import *
from modules.pipeline import Pipeline

class BertFCPipeline(Pipeline):
    # Pipeline example for text cnn classification model using THUCNews Data
    """
        Task:   Text Classification
        Model:  Bert + FC
    """
    def build_field(self, **kwargs):
        max_length=None
        if 'max_length' in kwargs:
            max_length = kwargs['max_length']

        news_txt_field = Field(name='text', tokenizer=None, seq_flag=True, bert_flag=True, max_length=max_length)
        label_filed = Field(name='label', tokenizer=None, seq_flag=False, is_target=True, categorical=True, expand_flag=False,
                            num_classes=self.num_classes)
        self.fields_dict = {"text": news_txt_field, "label": label_filed}
        self.vocab_group = [["text"]]
        self.bert_dict = self.bert_pretrained_path['vocab_path']

    def build_model(self):
        self.model = BertFC(num_classes=self.num_classes,
                            model_path=self.bert_pretrained_path['model_path'],
                            bert_config=BertConfig(self.bert_pretrained_path['dict_path']))

        self.model.summary()
        #opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        #self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

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

        if self.num_classes > 2:
            loss = tf.keras.losses.CategoricalCrossentropy()
        else:
            loss = tf.keras.losses.BinaryCrossentropy()

        opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        self.model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
        self.model.fit(self.train_iter.forfit(), steps_per_epoch=len(self.train_iter), epochs=epochs, \
                            validation_data=self.dev_iter.forfit(), validation_steps=len(self.dev_iter),
                            callbacks=callbacks)


class FastTextPipeline(Pipeline):
    # Pipeline example for fasttext classification model using THUCNews Data
    """
        Task:   Text Classification
        Model:  fasttext
    """
    def build_field(self, **kwargs):
        news_txt_field = Field(name='text', tokenizer=kwargs['tokenizer'], seq_flag=True)
        label_filed = Field(name='label', tokenizer=None, seq_flag=False, is_target=True, categorical=True, expand_flag=False,
                            num_classes=self.num_classes)
        self.fields_dict = {"text": news_txt_field, "label": label_filed}
        self.vocab_group = [["text"]]

    def build_model(self):
        self.model = FastText(vocab_size=len(self.fields_dict['text'].vocab),
                             embedding_dim=FastTextParas.embedding_dim,
                             num_classes=self.num_classes)
        self.model.summary()

    def train(self, epochs, callbacks):
        if self.num_classes > 2:
            loss = tf.keras.losses.CategoricalCrossentropy()
        else:
            loss = tf.keras.losses.BinaryCrossentropy()

        opt = tf.keras.optimizers.Adam(learning_rate=FastTextParas.learning_rate)
        self.model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
        self.model.fit_generator(self.train_iter.forfit(), steps_per_epoch=len(self.train_iter), epochs=epochs, \
                            validation_data=self.dev_iter.forfit(), validation_steps=len(self.dev_iter), callbacks=callbacks)


class TextCNNPipeline(Pipeline):
    # Pipeline example for text cnn classification model using THUCNews Data
    """
        Task:   Text Classification
        Model:  Text CNN (native Kim)
    """
    def build_field(self, **kwargs):
        news_txt_field = Field(name='text', tokenizer=kwargs['tokenizer'], seq_flag=True, fix_length=self.fix_length)
        label_filed = Field(name='label', tokenizer=None, seq_flag=False, is_target=True, expand_flag=False, categorical=True,
                            num_classes=self.num_classes)
        self.fields_dict = {"text": news_txt_field, "label": label_filed}
        self.vocab_group = [["text"]]

    def build_model(self):
        self.model = TextCNN(vocab_size=len(self.fields_dict['text'].vocab),
                             embedding_dim=TextCNNParas.embedding_dim,
                             num_classes=self.num_classes,
                             seq_len=self.fix_length,
                             filter_sizes=TextCNNParas.filter_sizes)

        ##TODO: 为模型添加默认parameter
        ##TODO: 添加固定词表大小功能

        ##TODO: use pretrained embeddings
        self.model.summary()

    def train(self, epochs, callbacks):
        """
         Build model, loss, optimizer and train
        :param epochs: number of epochs in training
        :param callbacks:
        :return:
        """
        # 二分类问题则选用 binary_cross_entropy
        if self.num_classes > 2:
            loss = tf.keras.losses.CategoricalCrossentropy()
        else:
            loss = tf.keras.losses.BinaryCrossentropy()

        opt = tf.keras.optimizers.Adam(learning_rate=TextCNNParas.learning_rate)
        self.model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
        self.model.fit(self.train_iter.forfit(), steps_per_epoch=len(self.train_iter), epochs=epochs, \
                            validation_data=self.dev_iter.forfit(), validation_steps=len(self.dev_iter),
                            callbacks=callbacks)

