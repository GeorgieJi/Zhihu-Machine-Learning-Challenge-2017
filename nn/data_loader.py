#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from itertools import islice

from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors

CHAR_EMBEDDING_FILE = '../input/char_embedding.txt'
WORD_EMBEDDING_FILE = '../input/word_embedding.txt'
TRAIN_DATA_FILE = '../input/question_train_set_with_columns.txt'
TEST_DATA_FILE = '../input/question_eval_set_with_columns.txt'
TOPIC_INFO_FILE = '../input/topic_info.txt'
TRAIN_LABEL_FILE = '../input/question_topic_train_set.txt'
MAX_NB_CHARS = 10000
MAX_NB_WORDS = 400000
EMBEDDING_DIM = 256
max_seq_length_title_word = 25
max_seq_length_des_word = 100


def get_labels():
    # 尝试自己处理下father_labels.pkl
    f = open('../input/topic_info.txt')
    idx_list = []
    y_labels = []
    for line in islice(f, 1, None):
        aaa = line.strip().split('\t')
        idx_list.append(int(aaa[0]))
        y_labels.append([item for item in aaa[1].split(',')])
    f.close()
    pd.to_pickle(y_labels, '../father_labels.pkl')

    f = open(TRAIN_LABEL_FILE)
    idx_list = []
    y_labels = []
    for line in islice(f, 1, None):
        int_index, label_list = line.strip().split('\t')
        idx_list.append(int(int_index))
        y_labels.append([int(item) for item in label_list.split(',')])
    f.close()
    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(y_labels)
    train_y = mlb.transform(y_labels)
    pd.to_pickle(y_train, '../input/y.pkl')
    pd.to_pickle(mlb, '../input/mlb_y.pkl')
    train_y_father = pd.read_pickle('../father_labels.pkl')
    mlb_fa = MultiLabelBinarizer(sparse_output=True)
    mlb_fa.fit(train_y_father)
    train_y_father = mlb_fa.transform(train_y_father)
    pd.to_pickle(y_train_father, '../input/y_fa.pkl')
    pd.to_pickle(mlb_fa, '../input/mlb_y_fa.pkl')
    return train_y, train_y_father


# 做点处理， 因为语料有变化
def get_corpus(str_type='char'):
    texts = []
    cnt = 0
    if str_type == 'char':
        flag = 0
    else:
        flag = 1

    for str_path in [TRAIN_DATA_FILE, TEST_DATA_FILE]:
        f = open(str_path)
        for line in islice(f, 1, None):
            if cnt % 100000 == 0:
                print(cnt)
            terms = line.strip().split('\t')
            # 下面的join函数是将原来标题和描述信息中的','分隔信息改为以' '分隔信息
            if len(terms) == 5:
                if str_type == 'char':
                    texts.append(' '.join(terms[1+flag].split(',')))
                if str_type == 'char':
                    texts.append(' '.join(terms[3+flag].split(',')))
            elif len(terms) == 3:
                if str_type == 'char':
                    texts.append(' '.join(terms[1+flag].split(',')))
            else:
                continue
            cnt += 1
        f.close()
        print(len(texts))

        # # 再来添加word_title
        if str_type != 'char':
            str_path = (str_path.split('.'))[0]

            dat_title_word = pd.read_csv(str_path + '_title_word.csv')
            print(dat_title_word.shape)
            dat_title_word['title_word'] \
                = dat_title_word['title_word'].fillna('')
            title_word = dat_title_word['title_word'].tolist()
            for item in title_word:
                texts.append(' '.join(item.split(',')))

            dat_title_word = pd.read_csv(str_path + '_description_word.csv')
            print(dat_title_word.shape)
            dat_title_word['description_word'] \
                = dat_title_word['description_word'].fillna('')
            title_word = dat_title_word['description_word'].tolist()
            for item in title_word:
                texts.append(' '.join(item.split(',')))
    return texts


def get_tokenizer():
    texts_char = get_corpus(str_type='char')
    tok_char = Tokenizer(nb_words=MAX_NB_CHARS)
    tok_char.fit_on_texts(texts_char)
    pd.to_pickle(tok_char, '../input/tokenizer_char_10000.pkl')
    print('save token_char_10000')

    texts_word = get_corpus(str_type='word')
    tok_word = Tokenizer(nb_words=MAX_NB_WORDS)
    tok_word.fit_on_texts(texts_word)
    pd.to_pickle(tok_word, '../input/tokenizer_word_400000.pkl')
    print('save token_word_400000')
    return tok_char, tok_word


def pre_data(str_path=TEST_DATA_FILE):
    lis_index, title_char, title_word, des_char, des_word = [], [], [], [], []
    f = open(str_path)
    cnt = 0
    for line in islice(f, 1, None):
        if cnt % 100000 == 0:
            print(cnt)
        terms = line.strip().split('\t')
        lis_index.append(int(terms[0]))
        if len(terms) == 5:
            title_char.append(terms[1])
            title_word.append(terms[2])
            des_char.append(terms[3])
            des_word.append(terms[4])
        # =======================
        # 确认下是不是错位了！
        elif len(terms) == 3:
            title_char.append(terms[1])
            title_word.append(terms[2])
            des_word.append('')
            des_char.append('')
        else:
            title_char.append('')
            title_word.append('')
            des_char.append('')
            des_word.append('')
        # ========================
        cnt += 1
    f.close()
    print('SIZE:', len(lis_index))

    # 再来添加word_title
    str_path = (str_path.split('.'))[0]
    dat_title_word = pd.read_csv(str_path + '_title_word.csv')
    print(dat_title_word.shape)
    dat_title_word['title_word'] = dat_title_word['title_word'].fillna('')
    title_word = dat_title_word['title_word'].tolist()
    # 添加word_description
    dat_title_word = pd.read_csv(str_path + '_description_word.csv')
    print(dat_title_word.shape)
    dat_title_word['description_word'] \
        = dat_title_word['description_word'].fillna('')
    des_word = dat_title_word['description_word'].tolist()
    return lis_index, title_char, title_word, des_char, des_word


if __name__ == '__main__':
    np.random.seed(128)
    tokenizer_char, tokenizer_word = get_tokenizer()
    y_train, y_train_father = get_labels()
    word2vec_char = KeyedVectors.load_word2vec_format(CHAR_EMBEDDING_FILE,
                                                      binary=False)
    print('Found %s word vectors of word2vec' % len(word2vec_char.vocab))
    word2vec_word = KeyedVectors.load_word2vec_format(WORD_EMBEDDING_FILE,
                                                      binary=False)
    print('Found %s word vectors of word2vec' % len(word2vec_word.vocab))
    print('Preparing embedding matrix')
    nb_words = min(MAX_NB_CHARS, len(tokenizer_char.word_index))+1
    embedding_matrix_char = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in tokenizer_char.word_index.items():
        if word in word2vec_char.vocab and i < nb_words:
            embedding_matrix_char[i] = word2vec_char.word_vec(word)
    print(embedding_matrix_char.shape)
    pd.to_pickle(embedding_matrix_char, '../input/embed_matrix_char.pkl')
    print('Null word embeddings: %d'
          % np.sum(np.sum(embedding_matrix_char, axis=1) == 0))
    print('Preparing embedding matrix')
    nb_words = min(MAX_NB_WORDS, len(tokenizer_word.word_index))+1
    embedding_matrix_word = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in tokenizer_word.word_index.items():
        if word in word2vec_word.vocab and i < nb_words:
            embedding_matrix_word[i] = word2vec_word.word_vec(word)
    print(embedding_matrix_word.shape)
    pd.to_pickle(embedding_matrix_word, '../input/embed_matrix_word.pkl')
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix_word,
                                                     axis=1) == 0))
    test_idx, _, test_title_word, _, test_des_word = pre_data(TEST_DATA_FILE)
    train_idx, _, train_title_word, _, train_des_word \
        = pre_data(TRAIN_DATA_FILE)
    train_title_word_input\
        = pad_sequences(tokenizer_word.texts_to_sequences(train_title_word),
                        maxlen=max_seq_length_title_word, padding='post')
    train_des_word_input\
        = pad_sequences(tokenizer_word.texts_to_sequences(train_des_word),
                        maxlen=max_seq_length_des_word, padding='post')
    test_title_word_input\
        = pad_sequences(tokenizer_word.texts_to_sequences(test_title_word),
                        maxlen=max_seq_length_title_word, padding='post')
    test_des_word_input\
        = pad_sequences(tokenizer_word.texts_to_sequences(test_des_word),
                        maxlen=max_seq_length_des_word, padding='post')
    print(train_title_word_input.shape, train_des_word_input.shape)
    idx = np.arange(y_train.shape[0])
    # 随机打乱的生成一个数字序列
    np.random.shuffle(idx)
    tr_id = idx[:int(y_train.shape[0]*0.9)]
    te_id = idx[int(y_train.shape[0]*0.9):]

    tr_title_word_input = train_title_word_input[tr_id]
    tr_des_word_input = train_des_word_input[tr_id]
    tr_y = y_train[tr_id]

    te_title_word_input = train_title_word_input[te_id]
    te_des_word_input = train_des_word_input[te_id]
    te_y = y_train[te_id]

    pd.to_pickle(tr_title_word_input, '../input/train_title_word.pkl')
    pd.to_pickle(te_title_word_input, '../input/valid_title_word.pkl')
    pd.to_pickle(tr_des_word_input, '../input/train_des_word.pkl')
    pd.to_pickle(te_des_word_input, '../input/valid_des_word.pkl')
    pd.to_pickle(tr_y, '../input/train_y.pkl')
    pd.to_pickle(te_y, '../input/valid_y.pkl')
    pd.to_pickle(test_title_word_input, '../input/test_title_word.pkl')
    pd.to_pickle(test_des_word_input, '../input/test_des_word.pkl')


def get_data(boo_use_fa=True):
    str_path = '../input/'
    str_train_file = ['train_title_word.pkl', 'train_des_word.pkl',
                      'train_y.pkl']
    str_valid_file = ['valid_title_word.pkl', 'valid_des_word.pkl',
                      'valid_y.pkl']
    str_test_file = ['test_title_word.pkl', 'test_des_word.pkl']
    str_embed_file = 'embed_matrix_word.pkl'
    str_mlb_file = 'mlb_y.pkl'    # 已经转换为多分类标签的结果
    if boo_use_fa:
        str_train_file.append('train_y_fa.pkl')
        str_valid_file.append('valid_y_fa.pkl')
    lis_train, lis_valid, lis_test = [], [], []
    # zip函数在这里把这两个向量组成的矩阵做了一个转置
    for ft, fv in zip(str_train_file, str_valid_file):
        lis_train.append(pd.read_pickle(str_path + ft))
        lis_valid.append(pd.read_pickle(str_path + fv))
    for fte in str_test_file:
        lis_test.append(pd.read_pickle(str_path + fte))
    embed_matrix = pd.read_pickle(str_path + str_embed_file)
    mlb_y = pd.read_pickle(str_path + str_mlb_file)
    int_vocab = MAX_NB_WORDS+1
    return [lis_train[:2], lis_train[2:]], [lis_valid[:2], lis_valid[2:]], \
        lis_test[:2], embed_matrix, mlb_y, int_vocab


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(int_index*batch_size, min(size, (int_index+1)*batch_size))
            for int_index in range(0, nb_batch)]


def batch_generator(train, y, batch_size=128, shuffle=True, use_fa=False):
    sample_size = train[0].shape[0]
    index_array = np.arange(sample_size)
    while 1:
        if shuffle:
            np.random.shuffle(index_array)
        # 得到每个batch里的起始和结束序列， 针对所有batch
        batches = make_batches(sample_size, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            x_batch_title = train[0][batch_ids]
            x_batch_des = train[-1][batch_ids]
            x_batch = [x_batch_title, x_batch_des]
            y_batch = y[0][batch_ids].toarray()
            if use_fa:
                y_batch_fa = y[-1][batch_ids].toarray()
                yield x_batch, [y_batch, y_batch_fa]
            else:
                yield x_batch, [y_batch]
