# -*- coding: utf-8 -*-
# 用于自定义训练时callback回调函数，定义提前结束，保存Summary等
import tensorflow as tf
from keras.callbacks import TensorBoard
import os

class BaseEvaluator(tf.keras.callbacks.Callback):

    pass

class Evaluator4Clf(tf.keras.callbacks.Callback):
    def __init__(self, model, valid_data, tsn):
        self.best_total_acc = 0.
        self.valid_data = valid_data
        self.model = model
        self.file_writer = tf.summary.create_file_writer(os.path.join())


    def on_epoch_end(self, epoch, logs=None):

        raise NotImplementedError

class Evaluator4Ner(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        raise NotImplementedError

class Evaluator4Seq2Seq(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        raise NotImplementedError

class Evaluator4Match(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        raise NotImplementedError