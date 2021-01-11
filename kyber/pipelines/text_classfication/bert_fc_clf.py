# -*- coding: utf-8 -*-
# Pipeline example for text cnn classification model using THUCNews Data
"""
    Task:   Text Classification
    Model:  Bert + FC (native Kim)
    Data:   THUCNews
"""
import os
import tensorflow as tf
from config import *
from utils import *
from models import BertFC
from modules.pipeline import Pipeline
from modules.evaluator import Evaluator4Clf
from tensorflow.python.keras.callbacks import TensorBoard

class BertFCPipeline(Pipeline):
    def build_field(self):
        vocab_path = "/Users/James/Study/pretrained_models/bert/chinese-bert_chinese_wwm_L-12_H-768_A-12/vocab.txt"
        # token_dict = load_vocab(vocab_path)  # 读取词典
        # tokenizer = Tokenizer(token_dict, do_lower_case=True)  # 建立临时分词器

        news_txt_field = Field(name='text', tokenizer=JiebaTokenizer, seq_flag=True, fix_length=512, bert_flag=True)
        label_filed = Field(name='label', tokenizer=None, seq_flag=False, is_target=True, categorical=True,
                            num_classes=10)
        self.fields_dict = {"text": news_txt_field, "label": label_filed}
        self.vocab_group = [["text"]]
        self.bert_dict = vocab_path

    def build_model(self):
        model_path = "/Users/James/Study/pretrained_models/bert/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_model.ckpt"
        dict_path = "/Users/James/Study/pretrained_models/bert/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_config.json"
        self.model = BertFC(num_classes=10, model_path=model_path, config_json=BertConfig(dict_path))

        # self.model.build((None, 512))
        self.model.summary()

    def train(self, epochs, callbacks):
        """
         Build model, loss, optimizer and train
        :param epochs: number of epochs in training
        :param callbacks:
        :return:
        """
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        print("start fit")
        self.model.fit(self.train_iter.forfit(), steps_per_epoch=len(self.train_iter), epochs=epochs, \
                            validation_data=self.dev_iter.forfit(), validation_steps=len(self.dev_iter),
                            callbacks=callbacks)



def train():
    bert_fc_pipeline = BertFCPipeline(raw_data=Config.thu_news_raw_data, \
                                        standard_data = os.path.join(Config.thu_news_standard_data,Config.standard_filename_clf),
                                        processor_cls=THUCNewsProcessor,
                                        dataloader_cls=ClassifierLoader)
    # 创建用于评估分类的回调函数
    evaluator = Evaluator4Clf(bert_fc_pipeline, Config.bert_fc_thucnews_log_path, Config.bert_fc_thucnews_model_path)
    tb_callback = TensorBoard(log_dir=Config.text_cnn_thucnews_log_path)

    bert_fc_pipeline.build(batch_size=16, data_refresh=True)
    bert_fc_pipeline.train(epochs=10, callbacks=[evaluator, tb_callback])
    bert_fc_pipeline.test()





def predict():
    text_cnn_pipeline = BertFCPipeline()
    text_cnn_pipeline.load_model(Config.text_cnn_thucnews_model_path, "best_model.weights")
    # text_cnn_pipeline.build()
    res = text_cnn_pipeline.inference(["我想去打篮球"],row_type="list")
    print(res.argmax())

if __name__ == '__main__':
    train()
    # predict()