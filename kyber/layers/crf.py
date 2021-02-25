# -*- coding: utf-8 -*-
# crf layer

import tensorflow as tf

class CRF(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CRF, self).__init__(**kwargs)
        pass

    def build(self, input_shape):
        target_size = input_shape[-1]
        super(CRF, self).build(input_shape)
        self.transitions = self.add_weight(
            name = "transitions",
            shape = (target_size, target_size),
            initializer="glorot_uniform",
            trainable=True
        )

    def _normal_func(self, hidden_states):
        raise NotImplementedError

    def _score_sentence(self, pred_states, tags):
        """
        计算目标路径得分
        :param pred_states: 序列标注分类的输出，映射到output分类[batch, seq_len, output_dim]
        :param tags:
        :return:
        """
        ## cite from 苏剑林 implementation

        emit_score = tf.einsum('bni, bni ->b', pred_states, tags)
        # 具体计算表示？
        trans_score = tf.einsum('bni, ij, bnj->b', tags[:, :-1], self.transitions, tags[:,1:])
        # 第一个乘法表示从第i个开始的trans分数，第二个內积表示第j个接受的过滤
        return emit_score + trans_score

    def log_norm_func(self):
        inputs_mask = inputs[:,:-1






















