#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Task:   Text Classification
    Model:  Text CNN (native Kim)
    Data:   THUCNews
"""
import os
from pipelines.text_clf import FastTextPipeline
from config import Config
from data_utils import THUCNewsProcessor, ClassifierLoader
from data_utils.tokenizer import JiebaTokenizer
from trainer import Trainer

def train():
    trainer = Trainer(raw_data=Config.thu_news_raw_data,
                      standard_data=os.path.join(Config.thu_news_standard_data,Config.standard_filename_clf),
                      processor_cls=THUCNewsProcessor,
                      dataloader_cls=ClassifierLoader,
                      pipeline_cls=FastTextPipeline,
                      log_path=Config.fasttext_thucnews_log_path,
                      model_save_path=Config.fasttext_thucnews_model_path,
                      tokenizer=JiebaTokenizer,
                      batch_size=32,
                      num_classes=10,
                      epochs=3)
    trainer.train()

def predict():
    pipeline = FastTextPipeline(num_classes=10)
    pipeline.load_model(Config.fasttext_thucnews_model_path, "best_model.weights")
    res = pipeline.inference(["我想去打篮球"],row_type="list")
    print(res.argmax())

if __name__ == '__main__':
    train()
    predict()
