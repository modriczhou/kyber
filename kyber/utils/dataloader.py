# -*- coding: utf-8 -*-

from utils.component import Example, Vocab
from collections import Counter
import numpy as np
from utils.generator import *
import tqdm

class BaseLoader(object):
    """
    Base class for data loader
    Load the preprocessed data in a standard format to batch iterator ready for model input.
    """
    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch_size = batch_size

    def batch_iter(self):
        pass

    def load4seq(self):
        raise NotImplementedError

class ClassifierLoader(BaseLoader):
    """
    Data loader for text classification.
    """
    def __init__(self, data_path: str, batch_size: int, fields, vocab_group):
        """
        :param fields: dict of fields for this data {field_name: filed_class}
        :param data_path: standard classification data file path
        :param vocab_group: groups of different fields that use the same vocab.
        """
        super(ClassifierLoader,self).__init__(data_path, batch_size)
        self._data_path = data_path
        self._fields = fields
        self._vocab2field = dict()
        self._field2vocab = dict()
        self._vocab_group = vocab_group
        self.vocabs = dict()
        self.batch_size = batch_size

        self._all_examples = []
        # self._steps = len(self._all_data[0]) // self.batch_size + int(len(self._all_data[0]) % self.batch_size != 0)
        self.map_dict_field()

    def map_dict_field(self):
        for i, group in enumerate(self._vocab_group):
            self._vocab2field[i] = group
            for field in group:
                assert field in self._fields
                self._field2vocab[field] = i

    def build_vocab(self):
        for vocab_key in self._vocab2field:
            vocab = Vocab(vocab_file=None, vocab_size=None, min_freq=1)
            vocab.fit_on_examples(self._all_examples, self._vocab2field[vocab_key])
            self.vocabs[vocab_key] = vocab
            # Set vocab for each field
            for field in self._vocab2field[vocab_key]:
                self._fields[field].set_vocab(vocab)
                print("field {} set".format(field))

        return self.vocabs

    def set_vocab(self, built_vocabs):
        self.vocabs = built_vocabs

    def load_data(self):
        """
        :return: 2d list of string:
                [[text1,text2,text3,...],
                 [label1,label2,label3,...]]
        """

        with open(self._data_path, 'r', encoding='utf-8') as f:
            self._all_examples = [Example.from_tsv(row, self._fields) for row in tqdm.tqdm(f.readlines())]

    # @classmethod
    def set_examples(self, examples):
        self._all_examples = examples

    def train_dev_split(self, train_ratio=0.8, dev_ratio=0.1):

        assert train_ratio + dev_ratio < 1
        np.random.shuffle(self._all_examples)
        cut1 = int(len(self._all_examples) * train_ratio)
        cut2 = int(len(self._all_examples) * (train_ratio + dev_ratio))

        train_loader = ClassifierLoader(self.data_path, self.batch_size, self._fields, self._vocab_group)
        dev_loader = ClassifierLoader(self.data_path, self.batch_size, self._fields, self._vocab_group)
        test_loader = ClassifierLoader(self.data_path, self.batch_size, self._fields, self._vocab_group)

        train_loader.set_examples(self._all_examples[:cut1])
        dev_loader.set_examples(self._all_examples[cut1:cut2])
        test_loader.set_examples(self._all_examples[cut2:])

        return train_loader, dev_loader, test_loader

    def build_iterator(self, tf_flag=False):
        generator = Generator4Clf(self._all_examples, self._fields, self.batch_size)
        # self.batch_per_epoch = len(self._all_examples) // self.batch_size + int(
        #     len(self._all_examples) % self.batch_size > 0)
        return generator
        # return generator.__iter__(tf_flag=tf_flag)

class SeqLoader(BaseLoader):
    """
    """
    def __init__():
        pass


class SeqLabelLoader(BaseLoader):
    """

    """
    def __init__():
        raise NotImplementedError