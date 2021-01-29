#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Task:   Text Classification
    Model:  Text CNN (native Kim)
    Data:   THUCNews
"""
import os
from pipelines.text_clf import TextCNNPipeline
from config import Config
from data_utils import THUCNewsProcessor, ClassifierLoader
from data_utils.tokenizer import JiebaTokenizer
from trainer import Trainer

def train():
    standard_data_dict = {"train":os.path.join(Config.thu_news_standard_data, Config.standard_filename_clf)}

    trainer = Trainer(raw_data=Config.thu_news_raw_data,
                      standard_data=standard_data_dict,
                      processor_cls=THUCNewsProcessor,
                      dataloader_cls=ClassifierLoader,
                      pipeline_cls=TextCNNPipeline,
                      log_path=Config.text_cnn_thucnews_log_path,
                      model_save_path=Config.text_cnn_thucnews_model_path,
                      tokenizer=JiebaTokenizer,
                      batch_size=32,
                      num_classes=10,
                      epochs=2,
                      fix_length=512,
                      data_refresh=False)
    trainer.train()

def predict():
    text_cnn_pipeline = TextCNNPipeline(fix_length=512, num_classes=10)
    text_cnn_pipeline.load_model(Config.text_cnn_thucnews_model_path, "best_model.weights")
    res = text_cnn_pipeline.inference(["我们才能"],row_type="list")
    print(res.argmax())

if __name__ == '__main__':
    train()
    # predict()

