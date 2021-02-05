# -*- coding: utf-8 -*-
# 用于自定义训练时callback回调函数，定义提前结束，保存Summary等
import tensorflow as tf
from keras.callbacks import TensorBoard
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as fscore
from config import Config

class BaseEvaluator(tf.keras.callbacks.Callback):
    pass

class Evaluator4Clf(tf.keras.callbacks.Callback):
    """
    Custom callback for classification task
    """
    def __init__(self, pipeline, log_path, model_path):
        super(Evaluator4Clf, self).__init__()
        self.best_f1 = 0.
        self.pipeline = pipeline
        # self.valid_data = self.pipeline.dev_iter.forfit()
        self.best_model_name = "best_model.weights"
        self.latest_model_name = "latest_checkpoint.weights"
        self.model_save_path = model_path
        if tf.__version__.startswith("2"):
            self.file_writer = tf.summary.create_file_writer(os.path.join(log_path, 'valid metrics'))
            self.file_writer.set_as_default()

    def on_epoch_end(self, epoch, logs=None):
        val_acc, precision, recall, f1 = self.evaluate(self.pipeline.dev_iter)
        if tf.__version__.startswith("2"):
            tf.summary.scalar("val_acc", data=val_acc, step=epoch)
            tf.summary.scalar("val_precision", data=precision, step=epoch)
            tf.summary.scalar("val_recall", data=recall, step=epoch)
            tf.summary.scalar("val_f1", data=f1, step=epoch)

        if f1>self.best_f1:
            self.best_f1 = f1
            self.pipeline.save(self.model_save_path, self.best_model_name, fields_save=True, weights_only=True)

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
    def __init__(self, pipeline, log_path, model_path):
        super(Evaluator4Ner, self).__init__()
        self.best_f1 = 0.
        self.pipeline = pipeline
        self.best_model_name = "best_model.weights"
        self.latest_model_name = "latest_checkpoint.weights"
        self.model_save_path = model_path
        if tf.__version__.startswith("2"):
            self.file_writer = tf.summary.create_file_writer(os.path.join(log_path, 'valid metrics'))
            self.file_writer.set_as_default()

    def on_epoch_end(self, epoch, logs=None):
        # 获得 "O"代表的id
        self.target_field = None
        for f, k in self.pipeline.fields_dict.items():
            if k.is_target:
                self.target_field = k
        if not self.target_field:
            print("Not target field found!")
            return

        val_acc, precision, recall, f1 = self.evaluate(self.pipeline.dev_iter)

        if tf.__version__.startswith("2"):
            tf.summary.scalar("val_acc", data=val_acc, step=epoch)
            tf.summary.scalar("val_precision", data=precision, step=epoch)
            tf.summary.scalar("val_recall", data=recall, step=epoch)
            tf.summary.scalar("val_f1", data=f1, step=epoch)

        if f1>self.best_f1:
            self.best_f1 = f1
            self.pipeline.save(self.model_save_path, self.best_model_name, fields_save=True, weights_only=True)

        print("val_acc: {:05f}, val_precision: {:05f}, val_recall: {:05f}, val_f1: {:05f}, best_f1: {:05f}".\
            format(val_acc, precision, recall, f1, self.best_f1))


    def evaluate(self, data):
        # valid_y, valid_true =[], []
        mention_predict, mention_right, mention_true, right, total = 0., 0., 0., 0., 0.
        for x_true, y_true in data:
            y_pred = self.model.predict(x_true).argmax(axis=-1)
            y_true = y_true.squeeze()
            mention_predict += (y_pred != self.target_field.vocab.word2id("O")).sum()
            mention_right += ((y_true == y_pred) & (y_pred != self.target_field.vocab.word2id("O"))).sum()
            mention_true = (y_true != self.target_field.vocab.word2id("O")).sum()
            right += (y_true == y_pred).sum()
            total += len(y_true.flatten())

        acc = right / total
        precision, recall = mention_right/mention_predict, mention_right/mention_true
        f1 = 2 * mention_right / (mention_predict + mention_true)
        return acc, precision, recall, f1


class Evaluator4Seq2Seq(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        raise NotImplementedError

class Evaluator4Match(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        raise NotImplementedError