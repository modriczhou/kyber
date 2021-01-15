# -*- coding: utf-8 -*-

from data_utils.component import *
from data_utils.bert_tokenizer import Tokenizer
from collections import Counter
import numpy as np
from data_utils.generator import *
import tqdm

class BaseLoader(object):
    """
    Base class for data loader
    Load the preprocessed data in a standard format to batch iterator ready for model input.
    """
    def __init__(self, batch_size: int, fields, vocab_group, bert_dict=None, sep='\t'):
        """
        :param fields: dict of fields for this data {field_name: filed_class}
        :param data_path: standard classification data file path
        :param vocab_group: groups of different fields that use the same vocab.
        """
        # self._data_path = data_path
        self._fields = fields
        self._vocab2field = dict()
        self._field2vocab = dict()
        self._vocab_group = vocab_group
        self.vocabs = dict()
        self.batch_size = batch_size
        self._bert_dict = bert_dict
        self._all_examples = []  # {"train":train_exp, "dev":dev_exp, "test":test_exp}, dev,test允许为None

        # self._steps = len(self._all_data[0]) // self.batch_size + int(len(self._all_data[0]) % self.batch_size != 0)
        self.map_dict_field()


    def map_dict_field(self):
        for i, group in enumerate(self._vocab_group):
            self._vocab2field[i] = group
            for field in group:
                assert field in self._fields
                self._field2vocab[field] = i

    def build_vocab(self, vocab_size=None, min_freq=1): # 后续可加入
        """
        :return:
        """
        for vocab_key in self._vocab2field:
            # print(self._vocab2field[vocab_key])
            if not self._fields[self._vocab2field[vocab_key][0]].bert_flag:
                vocab = Vocab(vocab_file=None, vocab_size=vocab_size, min_freq=min_freq)
                vocab.fit_on_examples(self._all_examples, self._vocab2field[vocab_key])
            else:
                vocab = BertVocab(self._bert_dict)
            self.vocabs[vocab_key] = vocab
            # Set vocab for each field
            for field in self._vocab2field[vocab_key]:
                self._fields[field].set_vocab(vocab)
                print("field {} set".format(field))
        return self.vocabs

    def set_vocab(self, built_vocabs):
        self.vocabs = built_vocabs

    def load_data(self, standard_data_dict):
        """
        :return: 2d list of string:
                [[text1,text2,text3,...],
                 [label1,label2,label3,...]]
        """
        ## TODO: 有些数据本身就分好了train，dev和test，可直接分开读取
        #
        # with open(self._data_path, 'r', encoding='utf-8') as f:
        #     self._all_examples = [Example.from_tsv(row, self._fields) for row in tqdm.tqdm(f.readlines())]
        ## standard_data_dict: a dict {"train":train_data_path, "dev":dev_data_path, "test":test_data_path}

        raise NotImplementedError

    # @classmethod
    def set_examples(self, examples):
        self._all_examples = examples

    def train_dev_split(self, examples_dict, train_ratio=0.8, dev_ratio=0.1):
        data_keys = examples_dict.keys()
        train_loader = ClassifierLoader(self.batch_size, self._fields, self._vocab_group, self._bert_dict)
        dev_loader = ClassifierLoader(self.batch_size, self._fields, self._vocab_group, self._bert_dict)
        test_loader = ClassifierLoader(self.batch_size, self._fields, self._vocab_group, self._bert_dict)

        for key, examples in examples_dict.items():
            np.random.shuffle(examples)

        if "train" in data_keys and "dev" not in data_keys and "test" not in data_keys:
            assert train_ratio + dev_ratio < 1
            cut1 = int(len(examples_dict['train']) * train_ratio)
            cut2 = int(len(examples_dict['train']) * (train_ratio + dev_ratio))
            train_loader.set_examples(examples_dict['train'][:cut1])
            dev_loader.set_examples(examples_dict['train'][cut1:cut2])
            test_loader.set_examples(examples_dict['train'][cut2:])
        elif "train" in data_keys and "test" in data_keys and "dev" not in data_keys:
            assert train_ratio<1
            np.random.shuffle(examples_dict['train'])
            cut1 = int(len(examples_dict['train']) * train_ratio)
            train_loader.set_examples(examples_dict['train'][:cut1])
            dev_loader.set_examples(examples_dict['train'][cut1:])
            test_loader.set_examples(examples_dict['test'])
        elif "train" in data_keys and "test" in data_keys and "dev" in data_keys:
            train_loader.set_examples(examples_dict['train'])
            dev_loader.set_examples(examples_dict['dev'])
            test_loader.set_examples(examples_dict['test'])

        elif "train" in data_keys and "test" not in data_keys and "dev" in data_keys:
            ## 截一半给测试集
            cut = int(len(examples_dict['dev']) * 0.5)
            train_loader.set_examples(examples_dict['train'])
            dev_loader.set_examples(examples_dict['dev'][:cut])
            test_loader.set_examples(examples_dict['dev'][cut:])
        else:
            print("Data loading missed")
            return
        return train_loader, dev_loader, test_loader

    def build_iterator(self, tf_flag=False):
        generator = Generator4Clf(self._all_examples, self._fields, self.batch_size)
        # self.batch_per_epoch = len(self._all_examples) // self.batch_size + int(
        #     len(self._all_examples) % self.batch_size > 0)
        return generator

class ClassifierLoader(BaseLoader):
    """
    Data loader for text classification.
    """
    def load_data(self, standard_data_dict):
        """
            return dict {"train":train_exp, "dev":dev_exp, "test":test_exp}, dev,test允许为None

        """
        ## TODO: 有些数据本身就分好了train，dev和test，可直接分开读取
        examples_dict = dict()
        for key, path in standard_data_dict.items():
            if path:
                with open(path, 'r', encoding='utf-8') as f:
                    examples_dict[key]= [Example.from_tsv(row, self._fields) for row in tqdm.tqdm(f.readlines())]
        return examples_dict

class SeqLabelLoader(BaseLoader):
    """
    Data loader for sequence labelling
    """
    def __init__(self, batch_size: int, fields, vocab_group, bert_dict=None, sep="\t"):
        """
        :param fields: dict of fields for this data {field_name: filed_class}
        :param data_path: standard classification data file path
        :param vocab_group: groups of different fields that use the same vocab.
        """
        self.sep = sep
        super(SeqLabelLoader, self).__init__(batch_size, fields, vocab_group, bert_dict)

    def load_data(self, standard_data_dict):
        ""
        examples_dict = dict()
        for key, path in standard_data_dict.items():
            if path:
                columns = []
                examples = []
                with open(path, 'r', encoding='utf-8') as f:
                    for row in f.readlines():
                        row = row.strip()
                        if row=="":
                            if columns:
                                examples.append(Example.from_list(columns, fields_dict=self._fields))
                        else:
                            for i, column in enumerate(row.split(self.sep)):
                                if len(columns)<i+1:
                                    columns.append([])
                                columns[i].append(column)
                    if columns:
                        examples.append(Example.from_list(columns, fields_dict=self._fields))

                examples_dict[key] = examples
        return examples_dict