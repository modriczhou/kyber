# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras.layers as layers
from kyber.modules.encoders import TransformerBlock
from kyber.layers.embeddings import BertEmbeddings
from kyber.modules.encoders.transformer_encoder import TransformerBlock
from tensorflow.python.keras import Input
from kyber.config import BertConfig

class BertEncoder(tf.keras.Model):
    def __init__(self, config, add_pooling_layer=True):
        super(BertEncoder, self).__init__()
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.initializer_range = config.initializer_range
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.intermediate_size = config.intermediate_size
        self.layer_norm_eps = config.layer_norm_eps

        self.initializer = tf.keras.initializers.TruncatedNormal(config.initializer_range)
        self.embeddings = BertEmbeddings(self.config, name="embeddings")

        self.hidden_layers = [TransformerBlock(embed_dim=self.hidden_size,
                                                num_heads=self.num_attention_heads,
                                                ff_dim=self.intermediate_size,
                                                rate=self.hidden_dropout_prob,
                                                name="layer_._{}".format(i)) for i in range(self.num_hidden_layers)]

        self.LayerNorm = layers.LayerNormalization(epsilon=self.layer_norm_eps, name="LayerNorm")

        self.dropout = layers.Dropout(self.hidden_dropout_prob)
        self.pooler = layers.Dense(units=self.hidden_size,
                                   kernel_initializer=self.initializer,
                                   activation='tanh',
                                   name='pooler')
        self.add_pooling_layer = add_pooling_layer

        # init with Model API to build
        inputs_ids = Input(shape=(None, ), dtype=tf.float32)
        type_ids = Input(shape=(None, ), dtype=tf.float32)
        inputs = [inputs_ids, type_ids]
        super(BertEncoder, self).__init__(inputs=inputs, outputs = self.call(inputs, training=False))


    def call(self, inputs, mask=None, **kwargs):
        embeddings = self.embeddings(inputs)
        # print(embeddings)
        embedds_layernorm = self.LayerNorm(embeddings)
        hidden_states = self.dropout(embedds_layernorm)

        for i, trans_block in enumerate(self.hidden_layers):
            hidden_states = trans_block(inputs=hidden_states, mask=mask)

        #outputs = hidden_states

        if self.add_pooling_layer:
            pooled_output = self.pooler(hidden_states[:,0])
            return hidden_states, pooled_output
        # print("pooled_output:", pooled_output)

        return hidden_states

    # def summary(self, *args):
    #     inputs_ids = Input(shape=(None, ), dtype=tf.float32)
    #     type_ids = Input(shape=(None, ), dtype=tf.float32)
    #     inputs = [inputs_ids, type_ids]
    #     tf.keras.Model(inputs=inputs, outputs=self.call(inputs, training=False)).summary()


    def variable_mapping(self, reference='bert'):
        """构建Keras层与checkpoint的变量名之间的映射表
        """
        mapping = [
            'bert/embeddings/word_embeddings',
            'bert/embeddings/token_type_embeddings',
        ]
        # if self.max_relative_position is None:
        mapping.extend(['bert/embeddings/position_embeddings'])

        mapping.extend([
            'bert/embeddings/LayerNorm/gamma',
            'bert/embeddings/LayerNorm/beta',
        ])

        # if self.embedding_size != self.hidden_size:
        #     mapping.extend([
        #         'bert/encoder/embedding_hidden_mapping_in/kernel',
        #         'bert/encoder/embedding_hidden_mapping_in/bias',
        #     ])

        # if reference == 'albert':
        #     block_weight_names = [
        #         'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel',
        #         'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias',
        #         'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel',
        #         'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias',
        #         'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel',
        #         'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias',
        #         'bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel',
        #         'bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias',
        #         'bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma',
        #         'bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta',
        #         'bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel',
        #         'bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias',
        #         'bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel',
        #         'bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias',
        #         'bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma',
        #         'bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta',
        #     ]=

        # if not self.block_sharing and reference != 'albert':
        for i in range(self.num_hidden_layers):
            block_name = 'layer_%d' % i
            mapping.extend([
                'bert/encoder/%s/attention/self/query/kernel' % block_name,
                'bert/encoder/%s/attention/self/query/bias' % block_name,
                'bert/encoder/%s/attention/self/key/kernel' % block_name,
                'bert/encoder/%s/attention/self/key/bias' % block_name,
                'bert/encoder/%s/attention/self/value/kernel' % block_name,
                'bert/encoder/%s/attention/self/value/bias' % block_name,
                'bert/encoder/%s/attention/output/dense/kernel' % block_name,
                'bert/encoder/%s/attention/output/dense/bias' % block_name,
                'bert/encoder/%s/attention/output/LayerNorm/gamma' % block_name,
                'bert/encoder/%s/attention/output/LayerNorm/beta' % block_name,
                'bert/encoder/%s/intermediate/dense/kernel' % block_name,
                'bert/encoder/%s/intermediate/dense/bias' % block_name,
                'bert/encoder/%s/output/dense/kernel' % block_name,
                'bert/encoder/%s/output/dense/bias' % block_name,
                'bert/encoder/%s/output/LayerNorm/gamma' % block_name,
                'bert/encoder/%s/output/LayerNorm/beta' % block_name,
            ])
        # elif not self.block_sharing and reference == 'albert':
        #     mapping.extend(block_weight_names * self.num_hidden_layers)
        # else:
        #     mapping.extend(block_weight_names)

        # if self.with_pool or self.with_nsp:
        if self.add_pooling_layer:
            mapping.extend([
                'bert/pooler/dense/kernel',
                'bert/pooler/dense/bias',
            ])
            # if self.with_nsp:
            #     mapping.extend([
            #         'cls/seq_relationship/output_weights',
            #         'cls/seq_relationship/output_bias',
            #     ])

        # if self.with_mlm:
        #     mapping.extend([
        #         'cls/predictions/transform/dense/kernel',
        #         'cls/predictions/transform/dense/bias',
        #         'cls/predictions/transform/LayerNorm/gamma',
        #         'cls/predictions/transform/LayerNorm/beta',
        #         'cls/predictions/output_bias',
        #     ])

        return mapping



    def load_weights_from_checkpoint(self,
                                     checkpoint_file,
                                     reference='bert',
                                     mapping=None):
        """从预训练好的Bert的checkpoint中加载权重
        """
        if mapping is None:
            mapping = self.variable_mapping(reference)
        # print(mapping)

        def load_variable(name):
            # 加载单个变量的函数
            variable = tf.train.load_variable(checkpoint_file, name)
            if name in [
                    'bert/embeddings/word_embeddings',
                    'cls/predictions/output_bias',
            ]:
                # if self.keep_words is None:
                return variable
            elif name == 'cls/seq_relationship/output_weights':
                return variable.T
            else:
                return variable

        values = [load_variable(name) for name in mapping]
        #print(len(mapping))
        #print(len(values))

        # bert_layer_names = [
        #     'embeddings',
        #     'Embedding-Segment',
        #     'Embedding-Position',
        #     'Embedding-Norm',
        #     'Embedding-Mapping',
        # ]
        #
        # for i in range(self.num_hidden_layers):
        #     bert_layer_names.extend([
        #         'Encoder-%d-MultiHeadSelfAttention' % (i + 1),
        #         'Encoder-%d-MultiHeadSelfAttention-Norm' % (i + 1),
        #         'Encoder-%d-FeedForward' % (i + 1),
        #         'Encoder-%d-FeedForward-Norm' % (i + 1),
        #     ])
        #
        # bert_layer_names.extend([
        #     'Pooler-Dense',
        #     'NSP-Proba',
        #     'MLM-Dense',
        #     'MLM-Norm',
        #     'MLM-Proba',
        # ])
        new_model_weights = []
        for layer in self.layers:
            # print(layer.name)
            # print(layer.weights)
            new_model_weights.extend(layer.weights)
        print(len(new_model_weights))

        # for i in range(len(mapping)):
        #     print(new_model_weights[i], mapping[i])

        if len(new_model_weights) != len(values):
            raise ValueError(
                'Expecting %s weights, but provide a list of %s weights.' %
                (len(new_model_weights), len(values)))

        K.batch_set_value(zip(new_model_weights, values))


if __name__ == '__main__':
    model_path = "/Users/James/Study/pretrained_models/bert/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_model.ckpt"
    config_dict = "/Users/James/Study/pretrained_models/bert/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_config.json"


    bert_config = BertConfig(config_dict)

    bert = BertEncoder(bert_config)
    bert.summary()
    # bert.build(input_shape=[(None,None),(None,None)])
    bert.load_weights_from_checkpoint(model_path)



    # print(bert.layers)
    # print(len(bert.layers))

    # bert.load_weights_from_checkpoint(model_path)
