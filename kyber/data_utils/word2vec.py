#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/18 09:38
# @Author  : Yuansheng Zhou
# @Site    : 
# @File    : word2vec.py
# @Software: PyCharm

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors
from config import *
import multiprocessing
import os

class Embedding(object):
	"""
	Convert id feature to embedding
	"""
	def __init__(self, field, embed_dim=50):
		self.field = field
		self.model_path = Config.embedding_model_prefix + self.field + '.kv'
		self.wv = None
		if os.path.exists(self.model_path):
			self.load_model()
		self.embed_dim = embed_dim

	def lookup(self, ids):
		return [self.wv[w] for w in ids]

	def train_model(self, sentences_file):
		model = Word2Vec(LineSentence(sentences_file), size=self.embed_dim, window=5, min_count=5, workers=multiprocessing.cpu_count())
		model.wv.save(Config.embedding_model_prefix + self.field + '.kv')

	def load_model(self):
		self.wv = KeyedVectors.load(Config.embedding_model_prefix + self.field + '.kv', mmap='r')