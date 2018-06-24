import pandas as pd
import numpy as np
import os
import sys


def pre_sample(src_path, question_path, mode='train', words_or_chars='words'):
    """
    words
    :param src_path:
    :param question_path:
    :param mode:
    :return:
    """
    src = pd.read_csv(src_path, delimiter=',')
    question = pd.read_csv(question_path, delimiter=',')
    src = pd.merge(src, question[['qid', words_or_chars]], how='left', left_on='q1', right_on='qid')
    src = pd.merge(src, question[['qid', words_or_chars]], how='left', left_on='q2', right_on='qid')

    # label	q1	q2	qid_x	words_x	qid_y	words_y
    save_txt_folder = '../../input/matchzoo/{0}_{1}'.format(mode, words_or_chars)
    save_txt_file_name = 'sample.txt'
    try:
        os.makedirs(save_txt_folder)
        print 'created folder: ', save_txt_folder
    except:
        pass
    save_txt_path = os.path.join(save_txt_folder, save_txt_file_name)
    if mode == 'train':
        select_column = ['label', words_or_chars+'_x', words_or_chars+'_y']
        src.loc[:, select_column].to_csv(path_or_buf=save_txt_path, sep='\t', index=False, header=False)
    else:
        src['label'] = 0
        select_column = ['label', words_or_chars + '_x', words_or_chars + '_y']
        src.loc[:, select_column].to_csv(path_or_buf=save_txt_path, sep='\t', index=False, header=False)


def pre_emb_file(emb_file_path, dict_path, output_path):
    """
    we have word_dict file like this: w07925 5319 \n w07924 228
    the final result should be like this:
    1278 0.110638 0.015863 -0.026429 ...
    :param src_path:
    :param question_path:
    :return:
    """
    emb_file = pd.read_csv(emb_file_path, delim_whitespace=True, index_col=0, header=None)
    dict_file = pd.read_csv(dict_path, delim_whitespace=True, index_col=0, header=None)
    emb_file.index = emb_file.index.str.lower()
    output_df = pd.merge(dict_file, emb_file, left_index=True, right_index=True)
    output_df.to_csv(path_or_buf=output_path, sep=' ', index=False, header=False)


def main(mode, words_or_chars):
    train_path = '../../input/train.csv' # label, q1, q2
    test_path = '../../input/test.csv' # q1, q2
    question_path = '../../input/question.csv' # qid, words, chars

    word_emb_path = '../../input/word_embed.txt' # Wxxx 0.1 0.2 ...
    char_emb_path = '../../input/char_embed.txt'

    # pre_sample(train_path, question_path, mode=mode, words_or_chars=words_or_chars)
    # notice: we need to remove header before running MatchZoo

    # mauually run test_preparation_for_classify.py. Then:

    basedir = '../../input/matchzoo/{0}_{1}'.format(mode, words_or_chars)
    dict_path = os.path.join(basedir, 'word_dict.txt')
    output_path = os.path.join(basedir, 'emb.txt')
    if words_or_chars == 'words':
        pre_emb_file(word_emb_path, dict_path=dict_path, output_path=output_path)
    else:
        pre_emb_file(char_emb_path, dict_path=dict_path, output_path=output_path)


def for_test():
    # put all samples into test
    pass

if __name__ == "__main__":
    for mode in ['test']:
        for words_or_chars in ['chars', 'words']:
            main(mode=mode, words_or_chars=words_or_chars)

# for test, we need first generate sample.txt, and then add one column for it.
