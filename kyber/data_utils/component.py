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
from kyber.data_utils.bert_tokenizer import Tokenizer
from collections import Counter
import numpy as np

class BertVocab(object):
    def __init__(self, bert_dict):
        print(bert_dict)
        self.tokenizer = Tokenizer(bert_dict, do_lower_case=True)
        self.PAD = self.tokenizer.token_to_id(self.tokenizer._token_pad)
        # self.max_length=max_length

    def text_to_ids(self, text, max_length=512, fix_length=None):
        """
        :param text:
        :return: [input_ids, token_type_ids]
        """
        ##TODO: 如何对list序列进行encode
        if isinstance(text, str):
            return self.tokenizer.encode(text, max_length=max_length, first_length=fix_length)
        elif isinstance(text, list):
            ##TODO: 是否需要添加[CLS],[SEP]在NER任务中
            token_ids = self.tokenizer.tokens_to_ids(text)

            return token_ids[:max_length], [0] * len(token_ids[:max_length])

        ## TODO: fix_length参数

class Example(object):
    @classmethod
    def from_tsv(cls, tsv_row, fields_dict, sep = '\t', label_flg=True):
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
                # TODO: 此处可把tokenize封装到一个函数里，并根据类型判断是否需要进行分词
        return example

    @classmethod
    def from_list(cls, list_row, fields_dict, label_flg=False):
        """
        :param list_row:
        :param fields_dict:
        :param target_flg: False，即为输入不包含target的部分
        :return:
        """
        example = cls()

        if not label_flg:
            filtered_fields_keys = [key for key in fields_dict.keys() if not fields_dict[key].is_target]
        else:
            filtered_fields_keys = fields_dict.keys()
        if len(list_row) == len(filtered_fields_keys):
            for i, key in enumerate(filtered_fields_keys):
                setattr(example, key, fields_dict[key].tokenize(list_row[i]))
        return example

    # @classmethod
    # def from_list(cls, list_seqs, fields_dict, label_flag=True):
    #     """
    #     :param list_seqs: list[seq], 每个column对应一个
    #     :fields_dict:
    #     :return:
    #     """
    #     example = cls()
    #     if not label_flag:
    #         filtered_fields_keys = [key for key in fields_dict.keys() if not fields_dict[key].is_target]
    #     else:
    #         filtered_fields_keys = fields_dict.keys()
    #
    #     if len(list_seqs) == len(filtered_fields_keys):
    #         for i, key in enumerate(filtered_fields_keys):
    #             setattr(example, key, fields_dict[key].tokenize(list_row[i]))
    #     return example

class Vocab(object):
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3
    ## TODO: 对于label为seq的数据，如何保证vocab.pad和该label应当pad的值相等，例如NER任务重，应当pad为"O"对应的词表id，但是不能保证为0；或者选定可以保留的reserved vocab


    def __init__(self, vocab_file=None, vocab_size=None, min_freq=1, reserved=True):
        """
        :param tokenizer: tokenizer based on " " or chinese tokenizer
        """
        self._word2id = {}
        # self._count = 0
        self.embeddings = None
        self._reserved = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] if reserved else []
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
            for field_tuple in field_list:
                words = getattr(ex, field_tuple[0])
                if not field_tuple[1].seq_flag:
                    words = [words]         # counter update需要iterable的对象
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

    def text_to_ids(self, seq_words, max_len=None):
        return [self.word2id(word) for word in seq_words][:max_len]

    def ids_to_text(self, seq_ids):
        return [self.id2word(id) for id in seq_ids]



class Field(object):
    def __init__(self, name, tokenizer, seq_flag, is_target=False, pad_first=False, categorical=False,
                 bert_flag=False,fix_length=None, num_classes=None, max_length=None, vocab_reserved=True, expand_flag=True):
        self.seq_flag = seq_flag            # 判断该field是否采用序列化tokenize
        self.vocab = None
        self.tokenizer = tokenizer
        self.is_target = is_target          # 标志是否为target，用于tf中生成训练数据，区分x,y
        self.name = name
        self.pad_first = pad_first
        self.categorical = categorical      # 是否为标签数值，需要转化为categorical的离散形式
        self.fix_length = fix_length
        self.num_classes = num_classes
        self.bert_flag = bert_flag          # 判断该field是否采用bert tokenizer进行处理
        self.max_length = max_length        # 最大长度，适用于bert tokenizer进行encode，vocab进行index转换
        self.vocab_reserved = vocab_reserved    # 词表中是否使用保留词，主要为了区分 将离散label转为id的情况
        self.expand_flag = expand_flag if (self.is_target and expand_flag is not None) else False  # 如果作为target，是否扩充维度(-1)，为了计算accuracy

    def set_vocab(self, vocab):
        self.vocab = vocab

    def build_vocab(self):
        """
        Field根据词表来直接build vocab
        :return:
        """
        raise NotImplementedError


    def tokenize(self, sentence):
        if self.seq_flag and not self.bert_flag:
            ## Bert可以不用分词，直接在后面进行encode
            if isinstance(sentence, str):
                # 只对str类型进行分词，如果已经是list[word1, word2..]即直接返回
                return self.tokenizer.tokenize(text=sentence)
        # if self.is_target:
        #     if isinstance(sentence, list):
        #         return [[s] for s in sentence]
        #     else:
        #         return [sentence]
        return sentence

    def texts_to_ids(self, seq_words, seq_len, max_len):
        if self.seq_flag:
            if self.vocab:
                ids_array = self.vocab.text_to_ids(seq_words, max_len)
                if not self.bert_flag:
                    # print(ids_array)
                    return [ids_array[:max_len] if max_len else ids_array]
                else:
                    # 对于bert tokenizer后的结果为[input_ids, token_type_ids]，所以要分别截断
                    return [ids[:max_len] if max_len else ids for ids in ids_array]

        # return self.vocab.text_to_ids()
        # return self.vocab.text_to_ids([seq_words], max_len)  #对整个field的值进行index转换

        return seq_words[:max_len] if max_len else seq_words

    def process_batch(self, batch_data, padding=True):
        ##TODO:其中进行encode部分可以抽象出来，考虑和Bert的encode方式进行统一封装;
        """
        :param batch_data: list[list[str]]
        :return:
        """
        seq_num = 2 if self.bert_flag else 1
        batch_res = [[] for _ in range(seq_num)]
        # #batch_res = [[]]
        padded_res = []
        if self.seq_flag:
            for seq in batch_data:
                # seq_num = np.array(seq).ndim
                seq_res = self.texts_to_ids(seq, seq_len=self.fix_length, max_len=self.max_length)
                for i in range(seq_num):
                    batch_res[i].append(seq_res[i])
            if padding:
                for i in range(seq_num):
                    padded_res.append(self.pad_sequences(batch_res[i], length=self.fix_length, padding=self.vocab.PAD))
                # padded_seqs = self.pad_sequences(batch_ids, length=self.fix_length, padding=self.vocab.PAD)
                return tuple(padded_res) if len(padded_res)>1 else padded_res[0]
            return tuple(batch_res) if len(batch_res)>1 else batch_res[0]
        # elif self.categorical:
        elif self.categorical:
            return tf.keras.utils.to_categorical(batch_data, num_classes=self.num_classes)
        else:
            return batch_data
        #TODO:这种似乎只能解决label的问题，对于input的部分无法记录cate和内容的对应关系，后面应该加入label encoder来解决


    def process_step(self, step_data, padding=True):
        """

        :param step_data: list[str]
        :param padding:
        :return:
        """
        seq_num =2 if self.bert_flag else 1
        step_res = []
        padded_res = []
        if self.seq_flag:
            seq_res = self.texts_to_ids(step_data, seq_len=self.fix_length, max_len=self.max_length)
            for i in range(seq_num):
                step_res.append(seq_res[i])
            if padding:
                for i in range(seq_num):
                    padded_res.append(self.pad_sequence(step_res[i], length=self.fix_length, padding=self.vocab.PAD))
                # padded_seq = self.pad_sequence(seq_res[], length=self.fix_length, padding=self.vocab.PAD)
                return tuple(padded_res) if len(padded_res)>1 else padded_res[0]
            return tuple(step_res) if len(step_res)>1 else step_res[0]
        elif self.categorical:
            return tf.keras.utils.to_categorical(step_data, num_classes=self.num_classes)
        return step_data
        # TODO:这种似乎只能解决label的问题，对于input的部分无法记录cate和内容的对应关系，后面应该加入label encoder来解决

    def pad_sequence(self, input_seq, length=None, padding=0):
        if not length:
            length = len(input_seq)
            # return np.array([input_seq])
        if not self.pad_first:
            padded = np.array([np.concatenate([input_seq, [padding] * (length - len(input_seq))]) \
                if len(input_seq)<length else input_seq[:length]])
        else:
            padded = np.array([np.concatenate([[padding] * (length - len(input_seq)), input_seq]) \
                if len(input_seq)<length else input_seq[:length]])
        return padded

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





