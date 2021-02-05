#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trainer class to select pipeline, control config and parameters
"""

from modules.evaluator import Evaluator4Clf
from tensorflow.python.keras.callbacks import TensorBoard

class Trainer(object):
    def __init__(self,
                 raw_data=None,              # Raw input data to feed into processor to get the standard data
                 standard_data=None,         # Standard format data for each task
                 processor_cls=None,         # Processor class e.g. THUCNewsProcessor
                 dataloader_cls=None,        # Data loader class e.g. ClassifierLoader
                 pipeline_cls=None,          # pipeline class e.g. TextCNNPipeline
                 log_path=None,              # folder path to save summary logs for TensorBoard
                 model_save_path=None,       # folder path to save model
                 tokenizer=None,        # tokenizer for each sequential text field
                 batch_size=32,
                 epochs=10,
                 num_classes=None,      # number of categories for text classification task
                 fix_length=None,
                 max_length=None,       # max length for bert model
                 data_refresh=False,
                 data_trec = 1.0,
                 evaluator = None,
                 bert_pretrained_path=None):

        self.raw_data = raw_data
        self.standard_data = standard_data
        self.processor_cls = processor_cls
        self.dataloader_cls = dataloader_cls
        self.pipeline_cls = pipeline_cls
        self.log_path = log_path
        self.model_save_path = model_save_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.data_refresh = data_refresh
        self.epochs = epochs
        self.bert_pretrained_path = bert_pretrained_path
        self.fix_length = fix_length
        self.max_length = max_length
        self.data_trec = data_trec
        # self.data_trec_none = 0.01
        self.evaluator_cls = evaluator
        print("trainer max_length:", self.max_length)


    def train(self):
        self.pipeline = self.pipeline_cls(raw_data=self.raw_data,
                                     standard_data_dict=self.standard_data,
                                     processor_cls=self.processor_cls,
                                     dataloader_cls=self.dataloader_cls,
                                     num_classes=self.num_classes,
                                     bert_pretrained_path=self.bert_pretrained_path,
                                     fix_length=self.fix_length,
                                     max_length=self.max_length,
                                     data_trec = self.data_trec)

        # evaluator = Evaluator4Clf(self.pipeline, self.log_path, self.model_save_path)
        tb_callback = TensorBoard(log_dir=self.log_path)
        callbacks = [tb_callback]
        if self.evaluator_cls:
            callbacks.append(self.evaluator_cls(self.pipeline, self.log_path, self.model_save_path))
        self.pipeline.build(tokenizer=self.tokenizer,
                       batch_size=self.batch_size,
                       data_refresh=self.data_refresh)
        self.pipeline.train(epochs=self.epochs, callbacks=callbacks)
        if not self.evaluator_cls:
            self.pipeline.save(self.model_save_path, "bert_model.weights", fields_save=True, weights_only=True)
        # self.pipeline.train(epochs=self.epochs, callbacks=[])
        self.pipeline.test()

    def inference(self, row):
        self.pipeline.inference(row, row_type="list")