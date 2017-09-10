#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras
import keras.backend as k
from keras.layers import Dense, Input, Dropout, TimeDistributed, \
    concatenate
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras import metrics


# 改进点： 数据清洗（有思路）； topic中， parent_id和topic_id的关系；
# topic_id的标题和描述信息怎么用上去（目前只想到怎么用到清洗中）； 加入RNN和LSTM，
# 以及一些变体结构；
# 尝试下其它非深度学习方法， 目前在着手贝叶斯; 能不能和推荐算法联系起来
def get_embedding(embedding_matrix, use_glove, vocab, embed_hidden_size,
                  train_embed, max_len):
    if use_glove:
        embed = Embedding(vocab, embed_hidden_size, weights=[embedding_matrix],
                          input_length=max_len, trainable=train_embed)
    else:
        embed = Embedding(vocab, embed_hidden_size, input_length=max_len)
    return embed


def base_model(embed, max_len_title, max_len_des, sent_hidden_size,
               activation, dp, l2_value, label_num, optimizer, mlp_layer=3,
               kind='max'):
    # max, sum, average
    print('Build model...')
    base_embeddings \
        = keras.layers.core.Lambda(lambda x: k.max(x, axis=1),
                                   output_shape=(sent_hidden_size,))
    if kind == 'sum':
        base_embeddings \
            = keras.layers.core.Lambda(lambda x: k.sum(x, axis=1),
                                       output_shape=(sent_hidden_size,))
    elif kind == 'max':
        # 原版本（sum）, 得分0.0047
        # base_embeddings \
        #     = keras.layers.core.Lambda(lambda x: K.sum(x, axis=1),
        #                                output_shape=(SENT_HIDDEN_SIZE,))
        # mean, L2 = 4 e-6时， 用mean， 得分0.0052; 用max， 得分0.0045 最好的
        base_embeddings \
            = keras.layers.core.Lambda(lambda x: k.max(x, axis=1),
                                       output_shape=(sent_hidden_size,))

    translate_title = TimeDistributed(Dense(sent_hidden_size,
                                            activation=activation))
    premise_title = Input(shape=(max_len_title,), dtype='int32')
    translate_des = TimeDistributed(Dense(sent_hidden_size,
                                          activation=activation))
    premise_des = Input(shape=(max_len_des,), dtype='int32')
    premise_title = embed(premise_title)
    premise_title = translate_title(premise_title)
    premise_title = base_embeddings(premise_title)
    premise_title = BatchNormalization()(premise_title)
    premise_des = embed(premise_des)
    premise_des = translate_des(premise_des)
    premise_des = base_embeddings(premise_des)
    premise_des = BatchNormalization()(premise_des)

    premise = concatenate([premise_title, premise_des], name="sentence_vector")
    joint = Dropout(dp)(premise)
    for i in range(mlp_layer):
        joint = Dense(2 * sent_hidden_size, activation=activation,
                      W_regularizer=l2(l2_value) if l2_value else None)(joint)
        joint = Dropout(dp)(joint)
        joint = BatchNormalization()(joint)
    # 全连接层 - 判别是不是属于某个类别
    predict = Dense(label_num, activation='sigmoid')(joint)
    model = Model(input=[premise_title, premise_des], output=predict)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=[metrics.top_k_categorical_accuracy])
    model.summary()
    return model
