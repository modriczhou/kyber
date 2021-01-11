#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/7 14:11
# @Author  : Yuansheng Zhou
# @Site    : 
# @File    : utils.py
# @Software: PyCharm

class dotdict(dict):
   """dot.notation access to dictionary attributes"""
   __getattr__ = dict.get
   __setattr__ = dict.__setitem__
   __delattr__ = dict.__delitem__


def is_string(s):
   """判断是否是字符串
   """
   return isinstance(s, str)
