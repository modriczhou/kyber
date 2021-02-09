#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Task:   Text Classification
    Model:  Rnn + Linear (Rnn 可选cell type: LSTM, GRU, SimpleRNN, 及Bidirectional)
    Data:   THUCNews
"""
import os
from kyber.pipelines.text_clf import RnnFCPipeline
from kyber.config import Config
from kyber.data_utils import THUCNewsProcessor, ClassifierLoader
from kyber.data_utils.tokenizer import JiebaTokenizer
from kyber.trainer import Trainer
from kyber.modules.evaluator import Evaluator4Clf

def train():
    standard_data_dict = {"train":os.path.join(Config.thu_news_standard_data, Config.standard_filename_clf)}

    trainer = Trainer(raw_data=Config.thu_news_raw_data,
                      standard_data=standard_data_dict,
                      processor_cls=THUCNewsProcessor,
                      dataloader_cls=ClassifierLoader,
                      pipeline_cls=RnnFCPipeline,
                      log_path=Config.rnn_fc_thucnews_log_path,
                      model_save_path=Config.rnn_fc_thucnews_model_path,
                      tokenizer=JiebaTokenizer,
                      batch_size=32,
                      num_classes=10,
                      epochs=5,
                      fix_length=None,
                      evaluator=Evaluator4Clf,
                      data_refresh=False,
                      vocab_size=5000,
                      min_freq=5)
    trainer.train()

def predict():
    rnn_fc_pipeline = RnnFCPipeline(fix_length=512, num_classes=10)
    rnn_fc_pipeline.load_model(Config.rnn_fc_thucnews_model_path, "best_model.weights")
    res = rnn_fc_pipeline.inference(["我们才能"],row_type="list")
    print(res.argmax())

if __name__ == '__main__':
    train()
    # predict()

