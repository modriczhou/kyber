# -*- coding: utf-8 -*-
# 用于自定义训练时callback回调函数，定义提前结束，保存Summary等
import tensorflow as tf
from keras.callbacks import TensorBoard
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as fscore
from kyber.config import Config

class BaseEvaluator(tf.keras.callbacks.Callback):

    pass

class Evaluator4Clf(tf.keras.callbacks.Callback):
    def __init__(self, pipeline, log_path):
        self.best_f1 = 0.
        self.pipeline = pipeline
        # self.valid_data = self.pipeline.dev_iter.forfit()
        self.best_model_name = "best_model.weights"
        self.latest_model_name = "latest_checkpoint.weights"
        self.file_writer = tf.summary.create_file_writer(os.path.join(log_path, 'valid metrics'))
        self.file_writer.set_as_default()


    def on_epoch_end(self, epoch, logs=None):
        val_acc, precision, recall, f1 = self.evaluate(self.pipeline.dev_iter)
        tf.summary.scalar("val_acc", data =val_acc, step=epoch)
        tf.summary.scalar("val_precision", data=precision, step=epoch)
        tf.summary.scalar("val_recall", data=recall, step=epoch)
        tf.summary.scalar("val_f1", data=f1, step=epoch)

        if f1>self.best_f1:
            self.best_f1 = f1
            self.pipeline.save(Config.text_cnn_thucnews_model_path, self.best_model_name, fields_save=True, weights_only=True)

        # print('val_acc: %.5f, precision: %.5f, recall: %.5f, f1: %.5f, best_f1: %.5f \n' %
        #       (val_acc, precision, recall, f1, self.best_f1))

    def evaluate(self, data):
        valid_y, valid_true =[], []
        for x_true, y_true in data:
            y_pred = self.model.predict(x_true).argmax(axis=1)
            y_true = np.argmax(y_true, axis=1)
            valid_y.extend(y_pred)
            valid_true.extend(y_true)
            # print(len(valid_y))
        valid_y, valid_true = np.array(valid_y), np.array(valid_true)

        right = (valid_true == valid_y).sum()
        acc = right/len(valid_y)
        res = fscore(valid_true, valid_y, average='macro')
        return acc, res[0], res[1], res[2]


class Evaluator4Ner(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        raise NotImplementedError

class Evaluator4Seq2Seq(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        raise NotImplementedError

class Evaluator4Match(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        raise NotImplementedError