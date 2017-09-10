import pandas as pd
import PyFeaturesValidation

BASE_DIR = 'input/'
TRAIN_DATA_FILE = BASE_DIR + 'train_set_softmax.csv'
TEST_DATA_FILE = BASE_DIR + 'question_eval_set_title_word.csv'


def voi_pick_out_train_with_onlt_one_topic():
    data = pd.read_csv('../input/question_topic_train_set.txt', delimiter='\t')
    print(data.shape)
    data = data.dropna(how='any', axis=1)
    print(data.shape)
    data = data.dropna(axis=0)
    print(data.shape)

    # 统计每个问题都会和多少个主题匹配,这个数目的分布情况
    # =======================================================
    data['topic_count'] = (data['topic_id'].apply(lambda x: len(x.split(','))))
    dist_train = data[(data['topic_count'] == 1)]
    dist_train = dist_train.drop(['topic_id', 'topic_count'], axis=1)
    print(dist_train.shape)
    PyFeaturesValidation.voi_generate_train(dist_train,
                                            str_class='train_only_one_topic')
    PyFeaturesValidation\
        .voi_transform_topic_id_to_softmax(str_class='train_only_one_topic')


def voi_pick_out_offline_test_with_five_topic():
    data = pd.read_csv('../input/question_topic_train_set.txt', delimiter='\t')
    data = data.dropna(axis=0)
    print(len(data))
    # 统计每个问题都会和多少个主题匹配,这个数目的分布情况
    # =======================================================
    dist_train = (data['topic_id'].apply(lambda x: len(x.split(','))))
    data['topic_count'] = dist_train
    dat_offline = data[(data['topic_count'] == 5)]
    dat_train = data[(data['topic_count'] != 5)]
    dat_train = dat_train.drop(['topic_id', 'topic_count'], axis=1)
    print(len(dat_train))
    PyFeaturesValidation.voi_generate_train(dat_train, str_class='train')
    PyFeaturesValidation.voi_transform_topic_id_to_softmax(str_class='train')
    dat_offline = dat_offline.drop(['topic_count'], axis=1)
    print(len(dat_offline))
    # 不同于train的处理， 因为不需要把主题展开
    data = pd.read_csv('../output/question_train_set_title_word.csv',
                       delimiter=',')
    data = pd.merge(dat_offline, data, on='question_id', how='left')
    data = data.dropna(axis=0)
    print(len(data))
    # print(data)
    data.to_csv('input/offline_test_set_origin_topic.csv', index=False,
                columns=['question_id', 'title_word', 'topic_id'])


def voi_only_one_label():
    data = pd.read_csv('input/train_set_softmax.csv')
    print(data)
    data = data.drop_duplicates('title_word_x')
    print(data.drop_duplicates('topic_id').shape)
    data.to_csv('input/train_set_softmax_all_one_label.csv', index=False)


voi_only_one_label()
