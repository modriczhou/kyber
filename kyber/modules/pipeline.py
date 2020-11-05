#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/28 18:31
# @Author  : Yuansheng Zhou
# @Site    : 
# @File    : pipeline.py
# @Software: PyCharm
import os
import tensorflow as tf

class Pipeline(object):
    def __init__(self, raw_data=None, standard_data=None, processor_cls=None, dataloader_cls=None):
        self.raw_data = raw_data
        self.standard_data = standard_data
        self.processor_cls = processor_cls
        self.dataloader_cls = dataloader_cls
        self.data_loader = None
        self.processor = None
        self.model = None
        self.fields_dict = None
        self.vocab_group = None

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

    def build_field(self):
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

        self.data_loader = self.dataloader_cls(self.standard_data, \
                                                batch_size=batch_size, fields=self.fields_dict, vocab_group=self.vocab_group)
        print("Start loading data: {}".format(self.standard_data))
        self.data_loader.load_data()
        print("Loader built and loading finished")

    def train_dev_split(self, train_ratio=0.8, dev_ratio=0.1):
        self.train_loader, self.dev_loader, self.test_loader = self.data_loader.train_dev_split(train_ratio, dev_ratio)

    def build_vocab(self):
        vocabs = self.train_loader.build_vocab()
        self.dev_loader.set_vocab(vocabs)
        self.test_loader.set_vocab(vocabs)

    def build_iter(self):
        # 构造生成器
        self.train_iter = self.train_loader.build_iterator(tf_flag=True)
        self.dev_iter = self.dev_loader.build_iterator(tf_flag=True)
        self.test_iter = self.test_loader.build_iterator(tf_flag=True)

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

    def build(self, data_refresh=False):
        self.process_data(refresh=data_refresh)
        self.build_field()
        self.build_loader(batch_size=32)
        self.train_dev_split()
        self.build_vocab()
        self.build_iter()
        self.build_model()

    def save_model(self, model_path, model_name, weights_only=True):
        """
        :param model_path: 保存model的path，须为folder
        :param weights_only: 是否为
        :return:
        """
        if weights_only:
            self.model.save_weights(os.path.join(model_path, model_name))
        else:
            self.model.save(os.path.join(model_path, model_name))

    def load_model(self, model_file, weights_only = True):
        if weights_only:
            if not self.model:
                self.build_model()
            self.model.load_weights(model_file)
        else:
            self.model = tf.keras.models.load_model(model_file)

    def test(self):
        """
        在test loader上测试效果
        :return:
        """
        self.model.evaluate_generator(self.test_iter.forfit())

    def inference(self, text):
        if self.model is None:
            print("Model not loaded")
            return
        return self.model.predict(text)
