# -*- coding: utf-8 -*-
# Pipeline example for text cnn classification model using THUCNews Data
"""
    Task:   Text Classification
    Model:  Text CNN (native Kim)
    Data:   THUCNews
"""
import tensorflow as tf
from config import *
from utils import *
from models import TextCNN
import os
from modules.pipeline import Pipeline

class TextCNNPipeline(Pipeline):
    def build_field(self):
        news_txt_field = Field(name='news_text', tokenizer=JiebaTokenizer, seq_flag=True, fix_length=512)
        label_filed = Field(name='label', tokenizer=None, seq_flag=False, is_target=True, categorical=True,
                            num_classes=10)
        self.fields_dict = {"news_text": news_txt_field, "label": label_filed}
        self.vocab_group = [["news_text"]]

    def build_model(self):
        self.model = TextCNN(vocab_size=len(self.fields_dict['news_text'].vocab),
                             embedding_dim=100,
                             num_classes=10,
                             seq_len=512)
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
        self.model.fit_generator(self.train_iter.forfit(), steps_per_epoch=len(self.train_iter), epochs=epochs, \
                            validation_data=self.dev_iter.forfit(), validation_steps=len(self.dev_iter), callbacks=callbacks)

def train():
    text_cnn_pipeline = TextCNNPipeline(raw_data=Config.thu_news_raw_data, \
                                        standard_data = os.path.join(Config.thu_news_standard_data,Config.standard_filename_clf),
                                        processor_cls=THUCNewsProcessor,
                                        dataloader_cls=ClassifierLoader)
    text_cnn_pipeline.build(data_refresh=True)
    text_cnn_pipeline.train(epochs=10, callbacks=[])
    model_name = "model_weights"
    text_cnn_pipeline.test()
    text_cnn_pipeline.save_model(Config.text_cnn_thunews_model_path, model_name, weights_only=True)

def predict():
    text_cnn_pipeline = TextCNNPipeline()
    text_cnn_pipeline.load_model(Config.text_cnn_thunews_model_path)
    text_cnn_pipeline.inference("sss")

if __name__ == '__main__':
    train()









