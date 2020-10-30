# -*- coding: utf-8 -*-
# 用于自定义训练时callback回调函数，定义提前结束，保存Summary等
import tensorflow as tf

class Evaluator4Clf(tf.keras.callbacks.Callback):
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



