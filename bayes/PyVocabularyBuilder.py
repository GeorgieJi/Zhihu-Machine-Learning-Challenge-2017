import codecs
import csv
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime


def word_match_share(row):
    q1words = {}
    for word in str(row['title_word']).lower().split(','):
        q1words[word] = 1
    return q1words


def voi_pro_term_topic_builder(lis_word_index):
    print(datetime.now())
    # 获得topic的索引
    # ========================
    data = pd.read_csv('input/train_set_origin_topic.csv', delimiter=',')
    lis_data = np.array(data['topic_id'].drop_duplicates()).tolist()
    # =========================
    dat_numerator = pd.DataFrame(np.zeros((len(lis_word_index), len(lis_data)),
                                          dtype=np.float16),
                                 index=lis_word_index, columns=lis_data)
    with codecs.open('../input/train_set_softmax_topic.csv',
                     encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for values in reader:
            for word in values[1].split(','):
                dat_numerator[int(values[3])][word] += 1
    dat_numerator.to_csv('/home/georgie/Desktop/dat_numerator.csv')
    print(datetime.now())


# 统计整个语料中有多少种不同的词出现(包括question和topic)
def voi_get_count_of_words():
    max_nb_words = 411720
    texts_1 = []
    with codecs.open('../input/train_set_softmax.csv', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for values in reader:
            texts_1.append(values[1])
    print('Found %s texts in train file' % len(texts_1))
    tokenizer = Tokenizer(num_words=max_nb_words)
    tokenizer.fit_on_texts(texts_1)
    # 统计在所有样本中总过出现了多少各不相同的词
    word_index = tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))


def voi_prob_topic_builder():
    data = pd.read_csv('../input/question_train_set_with_columns.txt',
                       delimiter='\t')
    # 匹配两个文件 - question_train_set 和 question_topic_train_set,
    # 再匹配两个文件 - (question_train_set 和 question_topic_train_set) 和 topic_info
    dat_question_topic_train_set \
        = pd.read_csv('../input/question_topic_train_set_multiline.txt',
                      delimiter='\t')
    dat_question_topic_info = pd.read_csv('../input/topic_info.txt',
                                          delimiter='\t')
    dat_question_plus_topic_train_set \
        = pd.merge(data, dat_question_topic_train_set, on='question_id',
                   how='left')
    dat_question_plus_topic_train_set \
        = pd.merge(dat_question_plus_topic_train_set, dat_question_topic_info,
                   on='topic_id', how='left')
    qids \
        = pd.Series(dat_question_plus_topic_train_set['title_word_y'].tolist())
    qid_topic_id \
        = pd.Series(dat_question_plus_topic_train_set['topic_id'].tolist())
    print(qid_topic_id.value_counts().to_frame(name=None))
    plt.figure(figsize=(12, 5))
    print(qids.value_counts())
    plt.hist(qids.value_counts(), bins=50)
    plt.yscale('log', nonposy='clip')
    plt.title('Occurancies of Every Topic')
    plt.xlabel('Number of occurences of question')
    plt.ylabel('Number of questions')
    plt.show()


# 目前只基于1-gram
def voi_prob_term_builder():
    print(datetime.now())
    # 测试版本： 加“_rip”
    data = pd.read_csv('../output/question_train_set_title_word.csv',
                       delimiter=',')
    data = data.dropna(how='any', axis=1)
    print(data.shape)
    obj_word_frequency = data.apply(word_match_share, axis=1, raw=True)
    dic_word_frquency = None
    for item in obj_word_frequency:
        dic_word_frquency = Counter(dic_word_frquency) + Counter(item)
    print(datetime.now())
    # 至此, 得到了每个term在训练集中的出现次数
    dat_word_frquency = pd.DataFrame.from_dict(dic_word_frquency,
                                               orient='index').reset_index()
    dat_word_frquency.to_csv('../output/dat_word_frquency.csv')
    # 样本数
    print(data.shape)
    print(datetime.now())
    # 得到prob_term
    for d, x in dic_word_frquency.items():
        dic_word_frquency[d] = x / data.shape[0]
    print(dic_word_frquency)
    dat_word_probability = pd.DataFrame.from_dict(dic_word_frquency,
                                                  orient='index').reset_index()
    dat_word_probability.to_csv('../output/dat_word_probability.csv')

    # 编号， 便于prob_term_topic的建立(Dataframe的处理方式)
    # ====================================================
    lis_word_index = []
    for d, x in dic_word_frquency.items():
        lis_word_index.append(d)
    voi_pro_term_topic_builder(lis_word_index)
    # ====================================================
