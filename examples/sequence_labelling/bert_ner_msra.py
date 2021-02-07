#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/15 16:08
# @Author  : Yuansheng Zhou
# @Site    :
# @File    : bert_ner_msra.py
# @Software: PyCharm


import os
from pipelines.seq_labelling import BertTokenClfPipeline
from config import Config
from kyber.data_utils import SeqLabelLoader
from kyber.modules.evaluator import Evaluator4Ner
from kyber.trainer import Trainer
from data_utils.bert_tokenizer import *
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.keras import backend as K

model_path = "/Users/James/Study/pretrained_models/bert/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_model.ckpt"
dict_path = "/Users/James/Study/pretrained_models/bert/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_config.json"
vocab_path = "/Users/James/Study/pretrained_models/bert/chinese-bert_chinese_wwm_L-12_H-768_A-12/vocab.txt"

bert_pretrained_path = {
    'model_path': model_path,
    'dict_path': dict_path,
    'vocab_path': vocab_path
}

tags = ["LOC","ORG","PER"]
num_labels = 2 * len(tags)+1

def train():
    standard_data_dict = {"train":Config.msra_standard_data_train, "test":Config.msra_standard_data_test}
    trainer = Trainer(standard_data=standard_data_dict,
                      dataloader_cls=SeqLabelLoader,
                      pipeline_cls=BertTokenClfPipeline,
                      log_path=Config.bert_ner_msra_log_path,
                      model_save_path=Config.bert_ner_msra_model_path,
                      batch_size=16,
                      num_classes=num_labels,
                      bert_pretrained_path=bert_pretrained_path,
                      data_refresh=True,
                      data_trec = 0.01,
                      max_length=128,
                      evaluator=Evaluator4Ner,
                      epochs=5)
    trainer.train()
    ##TODO：num_labels如何和对标签构建的vocab size保持一致
#   trainer.pipeline.load_model(Config.bert_fc_thucnews_model_path, "best_model.weights")
    # res = trainer.pipeline.inference(["谭望嵩奥运后再吃红牌！20分钟天津陷入10人作战 　　新浪体育讯　北京时间10月22日19:30，2008中超联赛第23轮的比赛打响一场焦点战。领头羊山东鲁能主场迎战9轮不败的强敌天津康师傅队。"],row_type="list")
    # print("res:",res.argmax())

def predict():
    pipeline = BertTokenClfPipeline(bert_pretrained_path=bert_pretrained_path, num_classes=num_labels)
    pipeline.load_model(Config.bert_ner_msra_model_path, "bert_model.weights")
    test_sen1 = "我们藏品中有几十册为清华北大图书馆所未藏"
    test_sen2 = "我想去北京看毛主席"
    res = pipeline.inference([list(test_sen1)], row_type='list')

    print(list(test_sen1))
    print(res)
    # print("res:",res.argmax())

if __name__ == '__main__':
    #y_true = [2, 1]
    #y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
    #parse_categorical_accuracy(y_true, y_pred)
    #train()
    predict()



