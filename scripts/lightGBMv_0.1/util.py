# -*- coding: utf-8 -*-
import pandas as pd

class DataSet(object):
    train_path = '../../input/train.csv'
    test_path = '../../input/test.csv'
    question_path = '../../input/question.csv'
    word_embed = "../../input/word_embed.txt"
    char_embed = "../../input/char_embed.txt"
    @classmethod
    def load_train(cls):
        train_df = pd.read_csv(cls.train_path)
        return train_df
    @classmethod
    def load_test(cls):
        test_df = pd.read_csv(cls.test_path)
        return test_df
    @classmethod
    def load_questions(cls):
        question_df = pd.read_csv(cls.question_path)
        return question_df
    @classmethod
    def load_word_embed(cls):
        word_embed_df = pd.read_csv(cls.word_embed, delim_whitespace=True, index_col=0, header=None)
        cnt_column = word_embed_df.shape[1]
        columns = map(lambda x:"word2vec_w_"+str(x),range(cnt_column))
        word_embed_df.columns = columns
        return word_embed_df
    @classmethod
    def load_char_embed(cls):
        char_embed_df = pd.read_csv(cls.char_embed, delim_whitespace=True, index_col=0, header=None)
        columns = ["word2vec_c_"+str(i) for i in range(char_embed_df.shape[1])]
        char_embed_df.columns = columns
        return char_embed_df


if __name__ == "__main__":
    print DataSet.load_word_embed()