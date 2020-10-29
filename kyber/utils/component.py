#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/2 15:11
# @Author  : Yuansheng Zhou
# @Site    : 
# @File    : component.py
# @Software: PyCharm

import abc
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from collections import Counter
import numpy as np

class Example(object):
    @classmethod
    def from_tsv(cls, tsv_row, fields_dict):
        """
        :param tsv_row: data row separated by "\t"
        :param fields_dict: fields dict {field_name:field instance}
        :return:
        """
        example = cls()
        data_cols = tsv_row.strip().split("\t")
        assert len(data_cols)==len(fields_dict), "data row col name: " + str(len(data_cols)) + " fields dict:" + str(len(fields_dict))
        if len(data_cols) == len(fields_dict):
            for i, key in enumerate(fields_dict.keys()):
                setattr(example, key, fields_dict[key].tokenize(data_cols[i]))

        return example

class Vocab(object):
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3

    def __init__(self, vocab_file=None, vocab_size=None, min_freq=1):
        """
        :param tokenizer: tokenizer based on " " or chinese tokenizer
        """
        self._word2id = {}
        # self._count = 0
        self.embeddings = None
        self._reserved = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        self._id2word = self._reserved[:]
        self._word_counter = Counter()
        self.vocab_size = vocab_size
        self.min_freq = min_freq

        if vocab_file:
            self.read_vocab_file(vocab_file)

    def __len__(self):
        return len(self._id2word)

    def fit_on_examples(self, examples_list, field_list):
        """
        Construct word vocab properties based on list of text sequences.
        """
        for ex in examples_list:
            for field in field_list:
                words = getattr(ex, field)
                self.insert_words(words)
        self.build_vocab()

    def insert_words(self, words):
        # 似乎不需要在字典中直接添加，可以先更新counter，后面根据vocab_size和word_freq删减时再更新
        self._word_counter.update(words)

    def build_vocab(self):
        # if min_freq < 1 or vocab_size<
        sorted_words = sorted(((w,f) for (w,f) in self._word_counter.items()), reverse=True, key=lambda x:x[1])
        if self.vocab_size:
            sorted_words = sorted_words[:self.vocab_size]

        for word, freq in sorted_words:
            if freq < self.min_freq:
                break
            self._word2id[word] = len(self._id2word)
            self._id2word.append(word)

    def word2id(self, word):
        return self._word2id.get(word, self.UNK)

    def id2word(self, id):
        return self._id2word[id]

    def read_vocab_file(self, vocab_file):
        """
        Read vocab from word freq file
        :param vocab_file:
        :return:
        """
        with open(vocab_file, 'r', encoding='utf-8') as f:
            all_words = f.readlines()

        raise NotImplementedError

    def load_embedding(self):
        raise NotImplementedError

    def text_to_ids(self, seq_words):
        return [self.word2id(word) for word in seq_words]

    def ids_to_text(self, seq_ids):
        return [self.id2word(id) for id in seq_ids]

class Batch(object):
    def __init__(self, batch_data, fields):
        self.name = None
        if batch_data is not None:
            self.batch_size = len(batch_data)
            self.fields = fields
            self.input_fields = [k for k, v in fields.items() if v is not None and not v.is_target]
            self.target_fields = [k for k, v in fields.items() if v is not None and v.is_target]

            for name, field in fields.items():
                if field is not None:
                    field_batch = [getattr(x, name) for x in batch_data]
                    setattr(self, name, field.process_batch(field_batch))

class Field(object):
    def __init__(self, name, tokenizer, seq_flag, is_target=False, pad_first=False, categorical=False, fix_length=None, num_classes=None):
        self.seq_flag = seq_flag
        self.vocab = None
        self.tokenizer = tokenizer
        self.is_target = is_target
        self.name = name
        self.pad_first = pad_first
        self.categorical = categorical
        self.fix_length = fix_length
        self.num_classes = num_classes

    def set_vocab(self, vocab):
        self.vocab = vocab

    def tokenize(self, sentence):
        if self.seq_flag:
            return self.tokenizer.tokenize(text=sentence)
        return sentence

    def texts_to_ids(self, seq_words, seq_len):
        if self.seq_flag:
            if self.vocab:
                ids_array = self.vocab.text_to_ids(seq_words)
                # print(ids_array)
                return ids_array[:seq_len] if seq_len else ids_array
        return seq_words[:seq_len] if seq_len else seq_words

    def process_batch(self, batch_data):
        """
        :param batch_data: list[list[str]]
        :return:
        """
        if self.seq_flag:
            batch_ids = [self.texts_to_ids(seq, seq_len=self.fix_length) for seq in batch_data]
            padded_seqs = self.pad_sequences(batch_ids, length=self.fix_length, padding=self.vocab.PAD)
            return padded_seqs
        else:
            return tf.keras.utils.to_categorical(batch_data, num_classes=self.num_classes)

    def pad_sequences(self, input_seqs, length=None, padding=0):
        if not length:
            length = max([len(seq) for seq in input_seqs])

        if not self.pad_first:
            padded = np.array([
                np.concatenate([x, [padding] * (length - len(x))])
                if len(x) < length else x[:length] for x in input_seqs
            ])

        else:
            padded = np.array([
                np.concatenate([[padding] * (length - len(x)), x])
                if len(x) < length else x[:length] for x in input_seqs
            ])

        return padded





