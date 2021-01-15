#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/28 18:31
# @Author  : Yuansheng Zhou
# @Site    : 
# @File    : pipeline.py
# @Software: PyCharm
import os
import tensorflow as tf
import pickle
from data_utils import Example, Step
import numpy as np
class Pipeline(object):
    def __init__(self, raw_data=None, standard_data=None, processor_cls=None, dataloader_cls=None, **kwargs):
        self.raw_data = raw_data
        self.standard_data = standard_data
        self.processor_cls = processor_cls
        self.dataloader_cls = dataloader_cls
        self.data_loader = None
        self.processor = None
        self.model = None
        self.fields_dict = None
        self.vocab_group = None
        self.bert_dict = None
        self.num_classes = None        # if it's a task for classification, specify the number of categories; also for the ner task

        if 'num_classes' in kwargs:
            self.num_classes = kwargs['num_classes']
        if 'bert_pretrained_path' in kwargs:
            self.bert_pretrained_path=kwargs['bert_pretrained_path']
        if 'fix_length' in kwargs:
            self.fix_length = kwargs['fix_length']
        if 'max_length' in kwargs:
            self.max_length=kwargs['max_length']
            print("pipeline max_length", self.max_length)

    def process_data(self, refresh):
        """
        :param processor: 进行文本预处理的Processor class，ex. THUNewsProcessor
        :param standard_data_path: 处理后的标准数据的路径
        :param standard_data_file: 处理后的标准数据的文件名
        """
        if self.raw_data is None and os.path.exists(self.standard_data):
            return
        else:
            self.processor = self.processor_cls(self.raw_data)
            refresh_flag = True if refresh else False
            self.processor.save_file(self.standard_data, refresh=refresh_flag)

        print("File saved at {}.".format(self.standard_data))

    def build_field(self, **kwargs):
        """
        子类自定义根据数据情况实现功能
        """
        raise NotImplementedError

    def build_loader(self, batch_size=64):
        """
        :param data_loader: 进行数据加载的DataLoader class, ex. ClassifierLoader
        :param standard_data: 预处理过的标准数据文件路径
        :param fields_dict: 标准数据中根据"\t"分割的各字段字典
        :param vocab_group: field的vocab共享分组
        :param batch_size: loader的batch size
        :return:
        """
        self.data_loader = self.dataloader_cls(batch_size=batch_size,
                                               fields=self.fields_dict,
                                               vocab_group=self.vocab_group,
                                               bert_dict=self.bert_dict)

        print("Start loading data: {}".format(self.standard_data))
        ## TODO: 可在直接创建dataloader时就完成数据读取init
        data_examples = self.data_loader.load_data(self.standard_data)
        print("Loader built and loading finished")
        return data_examples

    def train_dev_split(self, examples_dict, train_ratio=0.8, dev_ratio=0.1):
        self.train_loader, self.dev_loader, self.test_loader = self.data_loader.train_dev_split\
            (examples_dict, train_ratio, dev_ratio)

    def build_vocab(self):
        self.train_loader.build_vocab()
        # self.dev_loader.set_vocab(vocabs)
        # self.test_loader.set_vocab(vocabs)

    def build_iter(self):
        # 构造生成器
        self.train_iter = self.train_loader.build_iterator(tf_flag=True)
        self.dev_iter = self.dev_loader.build_iterator(tf_flag=True)
        self.test_iter = self.test_loader.build_iterator(tf_flag=True)
        # print(self.train_iter.)

    def build_model(self):
        """
        创建模型，子类自定义
        """
        raise NotImplementedError

    def train(self, epochs, callbacks):
        """
        :param epochs:
        :param callbacks:
        :return:
        定义训练，子类自定义
        """
        raise NotImplementedError

    def build(self, tokenizer, batch_size=32, data_refresh=False):
        self.process_data(refresh=data_refresh)
        self.build_field(tokenizer=tokenizer,
                         num_classes=self.num_classes,
                         max_length=self.max_length,
                         fix_length=self.fix_length)
        examples_dict = self.build_loader(batch_size=batch_size)
        self.train_dev_split(examples_dict)
        self.build_vocab()
        self.build_iter()
        self.build_model()

    def save(self, model_path, model_name, fields_save=True, weights_only=True):
        """
        保存model及在训练集上进行训练的vocab
        :param model_path: 保存model的path，须为folder
        :param weights_only: 是否为
        :return:
        """
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        if fields_save ==True:
            with open(os.path.join(model_path, "fields_dict.pkl"), 'wb') as f:
                 pickle.dump(self.fields_dict, f)
        if weights_only:
            self.model.save_weights(os.path.join(model_path, model_name))
        else:
            self.model.save(os.path.join(model_path, model_name), save_format='tf')

    def load_model(self, model_path, model_file, weights_only = True):
        with open(os.path.join(model_path, "fields_dict.pkl"),"rb") as f:
            self.fields_dict = pickle.load(f)
        if weights_only:
            if not self.model:
                self.build_model()
            self.model.load_weights(os.path.join(model_path, model_file))# .expect_partial()
        else:
            self.model = tf.keras.models.load_model(model_file)

    def test(self):
        """
        在test loader上测试效果
        :return:
        """
        self.model.evaluate(self.test_iter.forfit(), steps=len(self.test_iter))

    def inference(self, input, row_type="list"):
        """
        对单条文本进行预测
        :param text:
        :return:
        """
        if self.model is None:
            print("Model not loaded")
            return
        input_example=None
        if row_type=="list":
            input_example = Example.from_list(input, self.fields_dict)
        elif row_type=="tsv":
            input_example = Example.from_tsv(input, self.fields_dict)

        #print(input_example)

        if input_example:
            #print("s")
            step = Step(input_example, self.fields_dict)
            # print(step.step_x, len(step.step_x))
            return self.model.predict(step.step_x)[0]

        # return self.model.predict(text)

    def inference_batch(self):
        ## TODO: 编写用于批量预测的方法
        raise NotImplementedError