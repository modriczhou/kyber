#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/8 10:00
# @Author  : Yuansheng Zhou
# @Site    : 
# @File    : config.py
# @Software: PyCharm

class Config:
    thu_news_raw_data = "../test_data/raw_data/text_classification/THUCNews"
    thu_news_standard_data = "../test_data/standard/text_classification/THUCNews/"
    standard_filename_clf = "standard_clf_data.tsv"
    text_cnn_thucnews_model_path = "../saved_models/text_cnn_thucnews/"
    fasttext_thucnews_model_path = "../saved_models/fasttext_thucnews/"
    summary_log_path = "../summary_logs/"
    text_cnn_thucnews_log_path = "../summary_logs/text_cnn_thucnews/"
    fasttext_thucnews_log_path = "../summary_logs/fasttext_thucnews"


class TextCNNParas:
    pass
