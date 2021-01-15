#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

class dotdict(dict):
   """dot.notation access to dictionary attributes"""
   __getattr__ = dict.get
   __setattr__ = dict.__setitem__
   __delattr__ = dict.__delitem__


def is_string(s):
   """判断是否是字符串
   """
   return isinstance(s, str)


def shape_list(tensor: tf.Tensor):
   """
   用于解决tf1和tf2兼容问题，涵盖动态静态两种处理方式
   Deal with dynamic shape in tensorflow cleanly.
   Args:
       tensor (:obj:`tf.Tensor`): The tensor we want the shape of.
   Returns:
       :obj:`List[int]`: The shape of the tensor as a list.
   """
   dynamic = tf.shape(tensor)

   if tensor.shape == tf.TensorShape(None):
      return dynamic

   static = tensor.shape.as_list()

   return [dynamic[i] if s is None else s for i, s in enumerate(static)]