#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/10 17:26
# @Author  : Yuansheng Zhou
# @Site    : 
# @File    : text_cnn_clf.py
# @Software: PyCharm
import tensorflow as tf
from config import *
from utils import *
from models import TextCNN
import os
from modules.pipeline import Pipeline

class TextCNNPipeline(Pipeline):
    def build_field(self):
        news_txt_field = Field(name='news_text', tokenizer=JiebaTokenizer, seq_flag=True, fix_length=512)
        label_filed = Field(name='label', tokenizer=None, seq_flag=False, is_target=True, categorical=True,
                            num_classes=10)
        self.fields_dict = {"news_text": news_txt_field, "label": label_filed}
        self.vocab_group = [["news_text"]]

    def build_model(self):
        self.model = TextCNN(vocab_size=len(self.fields_dict['news_text'].vocab),
                             embedding_dim=100,
                             num_classes=10,
                             seq_len=512)
        self.model.summary()
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

if __name__ == '__main__':
    text_cnn_pipeline = TextCNNPipeline(raw_data=Config.thu_news_raw_data, \
                                        standard_data_path=Config.thu_news_standard_data,
                                        standard_data_file=Config.standard_filename_clf,
                                        processor_cls=THUNewsProcessor,
                                        dataloader_cls=ClassifierLoader)
    text_cnn_pipeline.process_data()
    text_cnn_pipeline.build_field()
    text_cnn_pipeline.build_loader(batch_size=32)
    text_cnn_pipeline.train_dev_split()
    text_cnn_pipeline.build_vocab()
    text_cnn_pipeline.build_iter()
    text_cnn_pipeline.build_model()
    text_cnn_pipeline.fit(epochs=20)








