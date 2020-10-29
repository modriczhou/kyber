#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/10 16:21
# @Author  : Yuansheng Zhou
# @Site    : 
# @File    : encoders.py
# @Software: PyCharm

# Base encoders Implementation
import tensorflow as tf
import tensorflow.keras.layers as layers

class BaseEcoder(layers.Layer):
    def __init__(self):
        super(BaseEcoder, self).__init__()
        raise NotImplementedError






