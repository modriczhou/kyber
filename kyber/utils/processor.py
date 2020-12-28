# -*- coding: utf-8 -*-

import os
import tqdm

class BaseProcessor(object):
    """
    Define base data processor class. Convert data to  
    """
    def __init__(self, data_path):
        self.data_path = data_path

    @staticmethod
    def open_file(file):
        return open(file, 'r', encoding='utf-8')
    
    @staticmethod
    def read4clf(cls):
        raise NotImplementedError

    def read4seq(cls):
        raise NotImplementedError

    # Save dataset for text classification to file.
    @staticmethod
    def write_to_clf(clf_data, save_file):
        """
        clf_data: List[List[str]] [[text1, label1],[text2,label2]...]
        file format: tsv, row: text + tab + label
        """
        with open(save_file, 'w', encoding='utf-8')as f:
            f.writelines("\n".join(["\t".join(str(r) for r in row) for row in clf_data]))
    
    @staticmethod
    def write_to_seq2seq(seq_data, save_file):
        """
        clf_data: List[List[str]] [[src1, tgt1],[src2,tgt2]...]
        file format: tsv, row: src + tab + tgt
        """
        with open(save_file, 'w', encoding='utf-8')as f:
            f.writelines("\n".join(["\t".join([str(r) for r in row]) for row in seq_data]))

    @staticmethod
    def write_to_ner(cls, ner_data, save_file):
        with open(save_file, 'w', encoding='utf-8')as f:
            f.writelines("\n".join(["\t".join(str(r) for r in row) for row in ner_data]))

class THUCNewsProcessor(BaseProcessor):
    """
    Process THUNews Dataset to a standard format for text classfication. 
    """
    def __init__(self, data_path):
        super(THUCNewsProcessor, self).__init__(data_path)
        self.categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
        self.cate2id = dict(zip(self.categories, range(len(self.categories))))

    def read4clf(self):
        """
        Read data from original dataset path, convert the categoriy to id and save the data in the format:
        List[List[str]]: [[text1, label1],[text2,label2]...]   
        """
        all_data = []
        for cate in self.categories:
            cate_path = str(os.path.join(self.data_path, cate))
            # print(cate_path)
            for cate_file in tqdm.tqdm(os.listdir(cate_path)[:200]):
                if cate_file.split('.')[-1]=="txt":
                    text_in = self.open_file(os.path.join(cate_path, cate_file)).read().\
                        replace("\n"," ").replace("\t", " ").strip()
                    if len(text_in):
                        all_data.append([text_in, self.cate2id[cate]])
        return all_data
    
    def save_file(self, standard_path, refresh=True):
        if not os.path.exists(standard_path) or refresh==True:
            all_data = self.read4clf()
            self.write_to_clf(all_data, standard_path)
