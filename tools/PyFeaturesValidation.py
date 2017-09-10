import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gensim.models import word2vec


def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['title_character_x']).lower().split(','):
            q1words[word] = 1
    for word in str(row['title_character_y']).lower().split(','):
            # print(word)
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0

    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    r = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words)
                                                             + len(q2words))
    return r


def int_word_match(str_path='../input/question_train_set_with_columns.txt',
                   str_split='\t'):
    data = pd.read_csv(str_path, delimiter=str_split)
    # 匹配两个文件 - question_train_set 和 question_topic_train_set,
    # 再匹配两个文件 - (question_train_set 和 question_topic_train_set)
    # 和 topic_info
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
    dat_question_plus_topic_train_set \
        = dat_question_plus_topic_train_set.dropna(axis=0)
    plt.figure(figsize=(15, 5))
    train_word_match \
        = dat_question_plus_topic_train_set.apply(word_match_share, axis=1,
                                                  raw=True)
    print(train_word_match.value_counts().max())
    return train_word_match.value_counts().max()


def voi_generate_train(dat_splitted_samples=None, str_class='train'):
    data = pd.read_csv('../output/question_train_set_title_word.csv',
                       delimiter=',')
    data = pd.merge(dat_splitted_samples, data, on='question_id', how='left')
    print(len(data))
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
    print(len(dat_question_plus_topic_train_set))
    dat_question_plus_topic_train_set.loc[:, {'question_id', 'title_word_x',
                                              'title_word_y', 'topic_id'}]\
        .to_csv('input/' + str_class + '_set_origin_topic.csv', index=False,
                columns=['question_id', 'title_word_x', 'title_word_y',
                         'topic_id'])


def fmap(a, b):
    return b, a


def voi_transform_topic_id_to_softmax(str_class='train'):
    data = pd.read_csv('input/train_set_origin_topic.csv', delimiter=',')
    lis_data = np.array(data['topic_id'].drop_duplicates()).tolist()
    print(len(lis_data))
    lik = range(0, len(lis_data))
    liv = list(lis_data)
    lim = map(fmap, lik, liv)
    d = dict(lim)
    print(d)
    print('=======================================================')
    data['topic_id'] = data['topic_id'].replace(d)
    data = data.dropna(axis=0)
    print(len(d))
    data.loc[:, {'question_id', 'title_word_x', 'topic_id'}] \
        .to_csv('input/' + str_class + '_set_softmax_topic.csv',
                index=False, columns=['question_id', 'title_word_x',
                                      'topic_id'])


model \
    = word2vec.KeyedVectors.load_word2vec_format("../input/word_embedding.txt",
                                                 binary=False)
print(model.most_similar(['w11', 'w54'], ['w6'], topn=3))
