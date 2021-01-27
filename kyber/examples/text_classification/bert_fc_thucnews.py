#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/12 15:28
# @Author  : Yuansheng Zhou
# @Site    : 
# @File    : bert_fc_thucnews.py.py
# @Software: PyCharm

import os
from pipelines.text_clf import BertFCPipeline
from config import Config
from data_utils import THUCNewsProcessor, ClassifierLoader
from trainer import Trainer

model_path = "/Users/James/Study/pretrained_models/bert/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_model.ckpt"
dict_path = "/Users/James/Study/pretrained_models/bert/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_config.json"
vocab_path = "/Users/James/Study/pretrained_models/bert/chinese-bert_chinese_wwm_L-12_H-768_A-12/vocab.txt"

bert_pretrained_path = {
    'model_path': model_path,
    'dict_path': dict_path,
    'vocab_path': vocab_path
}

def train():
    standard_data_dict = {"train":os.path.join(Config.thu_news_standard_data, Config.standard_filename_clf)}

    trainer = Trainer(raw_data=Config.thu_news_raw_data,
                      standard_data=standard_data_dict,
                      processor_cls=THUCNewsProcessor,
                      dataloader_cls=ClassifierLoader,
                      pipeline_cls=BertFCPipeline,
                      log_path=Config.bert_fc_thucnews_log_path,
                      model_save_path=Config.bert_fc_thucnews_model_path,
                      tokenizer=None,
                      batch_size=16,
                      num_classes=10,
                      bert_pretrained_path=bert_pretrained_path,
                      data_refresh=True,
                      max_length=128,
                      epochs=5)
    trainer.train()
    trainer.pipeline.load_model(Config.bert_fc_thucnews_model_path, "best_model.weights")
    res = trainer.pipeline.inference(["谭望嵩奥运后再吃红牌！20分钟天津陷入10人作战 　　新浪体育讯　北京时间10月22日19:30，2008中超联赛第23轮的比赛打响一场焦点战。领头羊山东鲁能主场迎战9轮不败的强敌天津康师傅队。"],row_type="list")
    print("res:",res.argmax())

def predict():
    pipeline = BertFCPipeline(bert_pretrained_path=bert_pretrained_path, num_classes=10)
    pipeline.load_model(Config.bert_fc_thucnews_model_path, "best_model.weights")
    res = pipeline.inference(["谭望嵩奥运后再吃红牌！20分钟天津陷入10人作战 　　新浪体育讯　北京时间10月22日19:30，2008中超联赛第23轮的比赛打响一场焦点战。领头羊山东鲁能主场迎战9轮不败的强敌天津康师傅队。"],row_type="list")
    print("res:",res.argmax())

if __name__ == '__main__':
    train()
    # predict()
