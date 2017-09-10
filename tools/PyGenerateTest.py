import pandas as pd


data = pd.read_csv('../input/question_eval_set_with_columns.txt',
                   delimiter='\t')
print(data.shape)
print(data.loc[:, {'question_id', 'title_word'}])
print(data.loc[:, {'question_id', 'title_word'}].shape)
data.loc[:, {'question_id', 'title_word'}].\
    to_csv('input/question_eval_set_title_word.csv', index=False,
           columns=['question_id', 'title_word'])
