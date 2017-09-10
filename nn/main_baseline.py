#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

from data_loader import get_data,batch_generator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from model_zoo import get_embedding, base_model


def get_test_id():
    f = open('../input/question_eval_set.txt', 'r')
    idx_list = []
    for line in f:
        idx = line.strip().split('\t')[0]
        idx_list.append(int(idx))
    f.close()
    return idx_list


def get_ans(predict, idd, label_map=None):
    predict = predict.argsort(axis=1)[:, ::-1][:, :5]
    ll = label_map.classes_
    ans = [[ll[item] for item in items] for items in predict]
    res = pd.DataFrame(ans)
    res.index = idd
    return res


def run():
    print('Loading data...')
    # training和validation里面都已经分别包含了x和y的部分
    training, validation, test, embedding_matrix,  label_map, vocab\
        = get_data(str_use_fa=False)
    # training[-1]就是training数组的最后一个元素(既y)， 【0】是第一个元素（既x）
    # 返回的tr_gen是一个数组，每一个元素就是一个batch
    tr_gen = batch_generator(training[0], training[-1], batch_size=256,
                             shuffle=True)
    te_gen = batch_generator(validation[0], validation[-1], batch_size=1024,
                             shuffle=False)
    print('VOCAB size:{}'.format(vocab))

    # Summation of word embeddings
    use_glove = False
    train_embed = True
    embed_hidden_size = 128
    sent_hidden_size = 256
    batch_size = 512
    max_len_title = 30
    max_len_des = 128
    dp = 0.2
    l2 = 4e-06
    activation = 'relu'
    optimizer = 'adam'

    print('Embed / Sent = {}, {}'.format(embed_hidden_size, sent_hidden_size))
    print('GloVe / Trainable Word Embeddings = {}, {}'.format(use_glove,
                                                              train_embed))

    label_num = len(label_map.classes_)
    bst_model_path = '../model/base_model_v1.hdf5'
    predict_path = '../res/base_model_v1.pkl'
    res_path = '../res/base_model_v1.csv'
    # 并没有用题目中给的词向量而是自己在训练这个词向量, 这点对结果的影响应该很大
    embed = get_embedding(embedding_matrix, use_glove, vocab,
                          embed_hidden_size, train_embed, None)
    # Relu的理解， Dropout的使用场景， Adam等梯度方法
    model = base_model(embed, max_len_title, max_len_des, sent_hidden_size,
                       activation, dp, l2, label_num, optimizer, MLP_LAYER=1,
                       kind='max')

    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True,
                                       save_weights_only=True)
    # 15次epoch自动early stop
    model.fit_generator(tr_gen,
                        steps_per_epoch=int(
                            training[0][0].shape[0]/batch_size)+1,
                        epochs=100, verbose=1, validation_data=te_gen,
                        validation_steps=int(
                            validation[0][0].shape[0]/batch_size)+1,
                        max_q_size=20, callbacks=[early_stopping,
                                                  model_checkpoint])
    print('load weights')
    model.load_weights(bst_model_path)
    predict = model.predict(test)
    pd.to_pickle(predict, predict_path)
    test_idx = get_test_id()
    ans = get_ans(predict, test_idx, label_map)
    ans.to_csv(res_path, index=True, header=False)


if __name__ == '__main__':
    run()
