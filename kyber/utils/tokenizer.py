#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/1 19:05
# @Author  : Yuansheng Zhou
# @Site    : 
# @File    : tokenizer.py
# @Software: PyCharm
import jieba

class Tokenizer(object):
    """
    Base class for tokenizer
    """
    def __init__(self):
        pass

    def tokenize(self, text):
        raise NotImplementedError

class JiebaTokenizer(Tokenizer):
    """
    Jieba Tokenizer
    """
    def __init__(self):
        super(JiebaTokenizer, self).__init__()

    @classmethod
    def tokenize(cls, text):
        return jieba.lcut(text.strip())


class SpaceTokenizer(Tokenizer):
    """
    Tokenzier based on space
    """
    def __init__(self):
        super().__init__()

    def tokenize(self, text):
        return text.strip().split()


class CharTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    @classmethod
    def tokenize(cls, text):
        return list(text.strip())

class BertTokenizer(Tokenizer):
    pass
