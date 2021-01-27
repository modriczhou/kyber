#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/15 16:08
# @Author  : Yuansheng Zhou
# @Site    : 
# @File    : bert_ner_msra.py
# @Software: PyCharm

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/12 15:28
# @Author  : Yuansheng Zhou
# @Site    :
# @File    : bert_fc_thucnews.py.py
# @Software: PyCharm

import os
from pipelines.seq_labelling import BertTokenClfPipeline
from config import Config
from data_utils import SeqLabelLoader
from trainer import Trainer
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
                      max_length=128,
                      epochs=5)
    trainer.train()
    ##TODO：num_labels如何和对标签构建的vocab size保持一致
#   trainer.pipeline.load_model(Config.bert_fc_thucnews_model_path, "best_model.weights")
    # res = trainer.pipeline.inference(["谭望嵩奥运后再吃红牌！20分钟天津陷入10人作战 　　新浪体育讯　北京时间10月22日19:30，2008中超联赛第23轮的比赛打响一场焦点战。领头羊山东鲁能主场迎战9轮不败的强敌天津康师傅队。"],row_type="list")
    # print("res:",res.argmax())

def predict():
    pipeline = BertTokenClfPipeline(bert_pretrained_path=bert_pretrained_path, num_classes=10)
    pipeline.load_model(Config.bert_fc_thucnews_model_path, "best_model.weights")
    res = pipeline.inference(["我想去北京"])
    print("res:",res.argmax())


def sparse_categorical_accuracy(y_true, y_pred):
  """Calculates how often predictions matches integer labels.
  Standalone usage:
   y_true = [2, 1]
   y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
   m = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
   assert m.shape == (2,)
   m.numpy()
  array([0., 1.], dtype=float32)
  You can provide logits of classes as `y_pred`, since argmax of
  logits and probabilities are same.
  Args:
    y_true: Integer ground truth values.
    y_pred: The prediction values.
  Returns:
    Sparse categorical accuracy values.
  """
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = ops.convert_to_tensor(y_true)
  y_pred_rank = y_pred.shape.ndims
  y_true_rank = y_true.shape.ndims
  # If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
  print(len(K.int_shape(y_true)), len(K.int_shape(y_pred)))
  if (y_true_rank is not None) and (y_pred_rank is not None) and (len(
      K.int_shape(y_true)) == len(K.int_shape(y_pred))):
      y_true = array_ops.squeeze(y_true, [-1])
  # y_pred = math_ops.argmax(y_pred, axis=-1)
  print(y_true)

  # If the predicted output and actual output types don't match, force cast them
  # to match.
  # if K.dtype(y_pred) != K.dtype(y_true):
  #   y_pred = math_ops.cast(y_pred, K.dtype(y_true))
  #
  # return math_ops.cast(math_ops.equal(y_true, y_pred), K.floatx())


if __name__ == '__main__':
    #y_true = [2, 1]
    #y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
    #sparse_categorical_accuracy(y_true, y_pred)
    train()
    # predict()

    # tokenizer = Tokenizer(vocab_path, do_lower_case=True)
    # inputs = "谭望嵩奥运后再吃红牌！20分钟天津陷入10人作战"
    # ids, segs = tokenizer.encode(inputs)
    # print(ids, len(ids))
    # print(segs, len(segs))
    # print(list(inputs), len(inputs))

