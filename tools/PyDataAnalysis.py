import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import PyFeaturesValidation
from sklearn.feature_extraction.text import CountVectorizer


# 在数据集中加一个列头
def voi_append_columns():
    with open('../input/question_eval_set.txt', 'r') as original:
        data = original.read()
        print("Finished Read File")
    with open('../input/question_eval_set_with_columns.txt', 'w') as modified:
        modified.write("question_id\ttitle_character\ttitle_word\t"
                       "description_character\tdescription_word\n" + data)


def voi_transform_question_topic_train_set_to_mulitiline():
    # 替换文本内容操作
    old_file = '../input/question_topic_train_set.txt'
    fopen = open(old_file, 'r')
    w_str = ""
    for line in fopen:
        lis_tuple = line.split('\t')
        lis_topic = lis_tuple[1].split(',')
        for item in lis_topic:
            w_str += (lis_tuple[0] + '\t' + item.replace('\n', '') + '\n')
    wopen = open('../input/question_topic_train_set_multiline.txt', 'w')
    wopen.write(w_str)
    fopen.close()
    wopen.close()


# 数据预处理的分析阶段
def voi_question_train_set_analysis():
    pal = sns.color_palette()
    data = pd.read_csv('../input/question_train_set_with_columns.txt',
                       delimiter='\t')
    data_test = pd.read_csv('../input/question_eval_set_with_columns.txt',
                            delimiter='\t')
    print(data_test.shape)
    data = data.dropna(axis=0)
    data_test = data_test.dropna(axis=1, how='any')
    data_test = data_test.dropna(axis=0)
    print(data_test.shape)

    # 统计每个问题标题由多少个单词组成的分布情况 - 训练集和测试集的分布很一致
    # ==================================
    dist_train = (data['title_word'].apply(lambda x: len(x.split(','))))
    dist_test = (data_test['title_word'].apply(lambda x: len(x.split(','))))
    plt.figure(figsize=(15, 10))
    plt.hist(dist_train, bins=50, range=[0, 50], color=pal[2], normed=True,
             label='train')
    plt.hist(dist_test, bins=50, range=[0, 50], color=pal[1], normed=True,
             alpha=0.5, label='test')
    plt.title('Normalised histogram of word count in each question title_word',
              fontsize=15)
    plt.legend()
    plt.xlabel('Number of words', fontsize=15)
    plt.ylabel('Probability', fontsize=15)
    plt.show()
    # ==================================
    # # 统计每个问题描述由多少个单词组成的分布情况性- 训练集和测试集的分布很一致
    # ==================================
    dist_train = (data['description_word'].dropna(axis=0)
                  .apply(lambda x: len(x.split(','))))
    dist_test = (data_test['description_word'].dropna(axis=0)
                 .apply(lambda x: len(x.split(','))))
    plt.figure(figsize=(15, 10))
    plt.hist(dist_train, bins=50, range=[0, 50], color=pal[2], normed=True,
             label='train')
    plt.hist(dist_test, bins=50, range=[0, 50], color=pal[1], normed=True,
             alpha=0.5, label='test')
    plt.title('Normalised histogram of word count in each question '
              'description_word', fontsize=15)
    plt.legend()
    plt.xlabel('Number of words', fontsize=15)
    plt.ylabel('Probability', fontsize=15)
    plt.show()
    # ==================================
    # 统计问题重复标题现象的分布(标题内容一模一样) - 有噪点, 需要再观察,
    # 是否这些标题重复的样本, 标签是否一样
    # ====================================
    dat_question_topic_train_set \
        = pd.read_csv('../input/question_train_set_with_columns.txt',
                      delimiter='\t')
    qids = pd.Series(dat_question_topic_train_set['title_word'].tolist())
    plt.figure(figsize=(12, 5))
    print(qids.value_counts())
    plt.hist(qids.value_counts(), bins=50)
    plt.yscale('log', nonposy='clip')
    plt.title('Log-Histogram of question appearance counts')
    plt.xlabel('Number of occurences of question')
    plt.ylabel('Number of questions')
    plt.show()
    # ====================================
    # 统计问题重复描述现象的分布(描述内容一模一样) - 有噪点, 需要再观察,
    # 是否这些描述重复的样本, 标签是否一样
    # ====================================
    dat_question_topic_train_set \
        = pd.read_csv('../input/question_train_set_with_columns.txt',
                      delimiter='\t')
    qids = pd.Series(dat_question_topic_train_set['description_word'].tolist())
    plt.figure(figsize=(12, 5))
    print(qids.value_counts())
    plt.hist(qids.value_counts(), bins=50)
    plt.yscale('log', nonposy='clip')
    plt.title('Log-Histogram of question appearance counts')
    plt.xlabel('Number of occurences of question')
    plt.ylabel('Number of questions')
    plt.show()
    # ===================================
    # 统计主题重复标题现象的分布(标题内容一模一样) - 无任何噪点
    # ====================================
    dat_question_topic_train_set = pd.read_csv('../input/topic_info.txt',
                                               delimiter='\t')
    qids = pd.Series(dat_question_topic_train_set['title_word'].tolist())
    plt.figure(figsize=(12, 5))
    print(qids.value_counts())
    plt.hist(qids.value_counts(), bins=50)
    plt.yscale('log', nonposy='clip')
    plt.title('Log-Histogram of question appearance counts')
    plt.xlabel('Number of occurences of question')
    plt.ylabel('Number of questions')
    plt.show()
    # ====================================
    # 统计主题重复描述现象的分布(描述内容一模一样)  -  有噪点
    # ====================================
    dat_question_topic_train_set = pd.read_csv('../input/topic_info.txt',
                                               delimiter='\t')
    qids = pd.Series(dat_question_topic_train_set['description_word'].tolist())
    plt.figure(figsize=(12, 5))
    print(qids.value_counts())
    plt.hist(qids.value_counts(), bins=50)
    plt.yscale('log', nonposy='clip')
    plt.title('Log-Histogram of question appearance counts')
    plt.xlabel('Number of occurences of question')
    plt.ylabel('Number of questions')
    plt.show()
    # ===================================


def voi_question_topic_analysis():
    pal = sns.color_palette()
    data = pd.read_csv('../input/question_topic_train_set.txt',
                       delimiter='\t')
    print(data.shape)
    data = data.dropna(how='any', axis=1)
    print(data.shape)
    data = data.dropna(axis=0)
    print(data.shape)
    # 统计每个问题都会和多少个主题匹配,这个数目的分布情况
    # =======================================================
    dist_train = (data['topic_id'].apply(lambda x: len(x.split(','))))
    plt.figure(figsize=(15, 10))
    print(dist_train)
    plt.hist(dist_train, bins=50, range=[0, 50], color=pal[2], normed=True,
             label='train')
    plt.title('Normalised histogram of topic count in each question',
              fontsize=15)
    plt.legend()
    plt.xlabel('Number of topics', fontsize=15)
    plt.ylabel('Probability', fontsize=15)
    plt.show()
    # =======================================================
    # 统计每个主题出现次数
    # =======================================================
    data = pd.read_csv('../input/question_train_set_with_columns.txt',
                       delimiter='\t')
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
    qids \
        = pd.Series(dat_question_plus_topic_train_set['title_word_y'].tolist())
    qid_topic_id \
        = pd.Series(dat_question_plus_topic_train_set['topic_id'].tolist())
    print(qid_topic_id.value_counts().to_frame(name=None))
    plt.figure(figsize=(12, 5))
    plt.hist(qids.value_counts(), bins=50)
    plt.yscale('log', nonposy='clip')
    plt.title('Occurancies of Every Topic')
    plt.xlabel('Number of occurences of question')
    plt.ylabel('Number of questions')
    plt.show()
    # =======================================================
    dat_question_topic_info = pd.read_csv('../input/topic_info.txt',
                                          delimiter='\t')
    # 统计每个主题标题由多少个单词组成的分布情况 - 绝大多数主题标题就一个词
    # ==================================
    dist_train = (dat_question_topic_info['title_word']
                  .apply(lambda x: len(x.split(','))))
    plt.figure(figsize=(15, 10))
    plt.hist(dist_train, bins=50, range=[0, 50], color=pal[2], normed=True,
             label='train')
    plt.title('Normalised histogram of word count in each question title_word',
              fontsize=15)
    plt.legend()
    plt.xlabel('Number of words', fontsize=15)
    plt.ylabel('Probability', fontsize=15)
    plt.show()
    # ==================================
    # # 统计每个主题描述由多少个单词组成的分布情况性 - 绝大多数描述都两个词
    # ==================================
    dist_train = (dat_question_topic_info['description_word'].dropna(axis=0)
                  .apply(lambda x: len(x.split(','))))
    plt.figure(figsize=(15, 10))
    plt.hist(dist_train, bins=50, range=[0, 50], color=pal[2], normed=True,
             label='train')
    plt.title('Normalised histogram of word count in each question '
              'description_word', fontsize=15)
    plt.legend()
    plt.xlabel('Number of words', fontsize=15)
    plt.ylabel('Probability', fontsize=15)
    plt.show()
    # ==================================


def voi_analysis_stop_word(str_column='title_character', str_level='word'):
    # 监视数据
    # 测试用小数据集
    data = pd.read_csv('../input/question_train_set_with_columns.txt',
                       delimiter='\t', usecols=['question_id', str_column])
    # # 打印出存在NaN值的行
    data = data.dropna(axis=0)
    print(data[data.isnull().values])
    for column in [str_column]:
        # 获得词频
        vectorizer = CountVectorizer()
        x = vectorizer.fit_transform(data[column])
        word = vectorizer.get_feature_names()
        print(word)
        f = open('output/Frequency_for_Every_' + str_level + '_Column_'
                 + column + '.csv', "w")
        for item in word:
            f.write(item + '\n')
        f.close()
        numpy.savetxt('output/Frequency_for_Every_'
                      + str_level + '_' + column + '.csv', x.sum(axis=0),
                      delimiter='\n', fmt='%f')


# 常用中文停止词有28个(也有地方说有800多个), 常用标点16种, 这里删除了390,
# 也就是把frequency在1W以上的当作是stop word
def voi_cut_stop_word(str_class='eval', str_column='title_character'):
    # 监视数据
    # 测试用小数据集
    data = pd.read_csv('../input/question_' + str_class
                       + '_set_with_columns.txt',
                       delimiter='\t', usecols=['question_id', str_column])
    # 打印出存在NaN值的行
    print(data.shape)
    print(data[data.isnull().values])
    data.to_csv('output/question_' + str_class + '_set_' + str_column + '.csv',
                index=False)
    dat_word_list_order_by_frequency \
        = pd.read_csv('../output/Frequency_for_'
                      'Every_Character_with_Columns_title_character.csv')

    int_iteration = 925
    int_previous = 1
    for i in range(int_iteration):
        print(i, dat_word_list_order_by_frequency['word_id'][i])

        # 替换文本内容操作
        old_file = 'output/question_' + str_class \
                   + '_set_' + str_column + '.csv'
        fopen = open(old_file, 'r')
        w_str = ""
        for line in fopen:
            # print(line)
            line = \
                line.replace(','
                             + dat_word_list_order_by_frequency['word_id'][i]
                             + ',', ',')
            line = \
                line.replace(','
                             + dat_word_list_order_by_frequency['word_id'][i]
                             + '"', '"')
            line = \
                line.replace('"'
                             + dat_word_list_order_by_frequency['word_id'][i]
                             + ',', '"')
            w_str += line
        wopen = open('output/question_' + str_class + '_set_'
                     + str_column + '.csv', 'w')
        wopen.write(w_str)
        fopen.close()
        wopen.close()

        int_current \
            = PyFeaturesValidation.int_word_match('output/question_'
                                                  + str_class + '_set_'
                                                  + str_column + '.csv',
                                                  str_split=',')
        if int_current != int_previous:
            print('Notic! Word-'
                  + dat_word_list_order_by_frequency['word_id'][i],
                  (int_current-int_previous)/int_previous)

        int_previous = int_current


# 清洗规则, 应该是把所有只有停止词标题的样本去掉
def voi_analysis_of_duplication():
    data = pd.read_csv('../input/question_train_set_with_columns.txt',
                       delimiter='\t')
    dat_question_topic_train_set \
        = pd.read_csv('../input/question_topic_train_set.txt', delimiter='\t')
    dat_question_topic_info \
        = pd.read_csv('../input/topic_info.txt', delimiter='\t')
    dat_question_plus_topic_train_set \
        = pd.merge(data, dat_question_topic_train_set, on='question_id',
                   how='left')
    dat_question_plus_topic_train_set \
        = pd.merge(dat_question_plus_topic_train_set, dat_question_topic_info,
                   on='topic_id', how='left')
    qids \
        = pd.Series(dat_question_plus_topic_train_set['title_word_x'].tolist())

    plt.figure(figsize=(12, 5))
    print(qids.value_counts())
    plt.hist(qids.value_counts(), bins=50)
    plt.yscale('log', nonposy='clip')
    plt.title('Log-Histogram of question appearance counts')
    plt.xlabel('Number of occurences of question')
    plt.ylabel('Number of questions')
    plt.show()
