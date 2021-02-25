#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
from kyber.modules.pipeline import Pipeline
from kyber.data_utils import Field, Example, Step
from kyber.models import BertNer
from kyber.config import *

class BertTokenClfPipeline(Pipeline):
    # Pipeline example for bert for token classification
    """
        Task:   sequence labelling
        Model:  bert + fc
    """
    def build_field(self, **kwargs):
        max_length=None
        if 'max_length' in kwargs:
            max_length = kwargs['max_length']

        news_txt_field = Field(name='text',
                               tokenizer=kwargs['tokenizer'],
                               seq_flag=True,
                               bert_flag=True,
                               max_length=max_length)

        label_field = Field(name='label',
                            tokenizer=None,
                            seq_flag=True,
                            is_target=True,
                            vocab_reserved=True,
                            max_length=max_length,
                            expand_flag=True) # 为了使用sparse_category_acc

        self.fields_dict = {"text": news_txt_field, "label": label_field} ## 顺序和column一致
        self.vocab_group = [["text"],["label"]]
        self.bert_dict = self.bert_pretrained_path['vocab_path']

    def build_model(self, ):
        self.model = BertNer(num_labels=self.num_classes,
                            model_path=self.bert_pretrained_path['model_path'],
                            bert_config=BertConfig(self.bert_pretrained_path['dict_path']))
        self.model.summary()

    def train(self, epochs, callbacks):
        """
         Build model, loss, optimizer and train
        :param epochs: number of epochs in training
        :param callbacks:
        :return:
        """
        # opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        # self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        print("start fit")
        opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        self.model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.train_iter.forfit(), steps_per_epoch=len(self.train_iter), epochs=epochs, \
                            validation_data=self.dev_iter.forfit(), validation_steps=len(self.dev_iter),
                            callbacks=callbacks)

    def inference(self, input, row_type="list"):
        """
        重写序列标注的预测方法
        :param input:
        :param row_type:
        :return:
        """
        if self.model is None:
            print("Model not loaded")
            return
        input_example = None
        if row_type == "list":
            input_example = Example.from_list(input, self.fields_dict)
        elif row_type == "tsv":
            input_example = Example.from_tsv(input, self.fields_dict)

        # print(input_example)

        if input_example:
            # print("s")
            step = Step(input_example, self.fields_dict)
            # print(step.step_x, len(step.step_x))
            model_res = self.model.predict(step.step_x)[0]

            idxs = model_res.argmax(axis=-1)

            self.target_field = None
            for f, k in self.fields_dict.items():
                if k.is_target:
                    self.target_field = k
            if not self.target_field:
                print("Not target field found!")
                return
            return [self.target_field.vocab.id2word(id) for id in idxs]
        return None



