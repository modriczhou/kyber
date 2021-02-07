#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json

class Config:
    thu_news_raw_data = "test_data/raw_data/text_classification/THUCNews"
    thu_news_standard_data = "test_data/standard/text_classification/THUCNews/"
    msra_standard_data_train = "test_data/standard/sequence_labeling/msra_chinese/msra_train_bio.txt"
    msra_standard_data_test = "test_data/standard/sequence_labeling/msra_chinese/msra_test_bio.txt"

    standard_filename_clf = "standard_clf_data.tsv"

    text_cnn_thucnews_model_path = "saved_models/text_cnn_thucnews/"
    fasttext_thucnews_model_path = "saved_models/fasttext_thucnews/"
    bert_fc_thucnews_model_path = "saved_models/bert_fc_thucnews/"
    bert_ner_msra_model_path = "saved_models/bert_ner_msra/"

    summary_log_path = "summary_logs/"
    text_cnn_thucnews_log_path = "summary_logs/text_cnn_thucnews/"
    fasttext_thucnews_log_path = "summary_logs/fasttext_thucnews"
    bert_fc_thucnews_log_path = "summary_logs/bert_fc_thucnews/"
    bert_ner_msra_log_path = "summary_logs/bert_ner_msra/"


class BertConfig(object):
    '''
    Default config for Bert Encoder
    '''
    def __init__(self, config_json=None):
        self.attention_probs_dropout_prob = 0.1
        self.directionality = "bidi"
        self.hidden_act = "gelu",
        self.hidden_dropout_prob = 0.1,
        self.hidden_size = 768,
        self.initializer_range = 0.02,
        self.intermediate_size = 3072,
        self.max_position_embeddings = 512,
        self.num_attention_heads = 12,
        self.num_hidden_layers = 12,
        self.pooler_fc_size = 768,
        self.pooler_num_attention_heads = 12,
        self.pooler_num_fc_layers = 3,
        self.pooler_size_per_head = 128,
        self.pooler_type = "first_token_transform",
        self.type_vocab_size = 2,
        self.vocab_size = 21128
        self.layer_norm_eps = 1e-12

        if config_json:
            with open(config_json, 'r', encoding='utf-8') as f:
                bert_config = json.load(f)
                for key in bert_config:
                    setattr(self, key, bert_config[key])

class TextCNNParas:
    filter_sizes = [3,4,5]
    embedding_dim = 100
    fix_seq_length = 512
    learning_rate = 0.001

class FastTextParas:
    embedding_dim = 100
    learning_rate = 0.001

