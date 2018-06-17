from __future__ import division
import itertools
from glove import Corpus, Glove
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
from sklearn.metrics import pairwise
import time
import json


def read_corpus(src_file, words_or_chars):
    """

    :param src_file: question_df
    :param words_or_chars: 'words' or 'chars'
    :return:
    """
    delchars = [chr(c) for c in range(256)]
    delchars = [x for x in delchars if not x.isalnum()]
    delchars.remove(' ')
    delchars = ''.join(delchars)

    datafile = src_file[words_or_chars]

    for line in datafile:
        yield line.translate(None, delchars).split(' ')


def train_glove(target_group, glove_para, src_file, save_model_name):
    """
    example: train_glove(target_group='words', glove_para=glove_para_word)
    after save the mode, u can use it by : glove_ana = Glove.load('glove_words.model')
    :param target_group: 'words' or 'chars'
    :param glove_para: glove_para_word = {'window_size':4, 'no_components':300, 'learning_rate':0.05, 'no_epochs':2, 'parallelism':4}
    :return:
    """
    corpus_model = Corpus()
    corpus_model.fit(read_corpus(src_file=src_file, words_or_chars=target_group), window=glove_para['window_size']) #avg word size is 6 for each sentence
    corpus_model.save('corpus_model_{}.model'.format(target_group))
    print target_group
    print('Dict size: %s' % len(corpus_model.dictionary))
    print('Collocations: %s' % corpus_model.matrix.nnz)
    print('Training the GloVe model')

    glove = Glove(no_components=glove_para['no_components'], learning_rate=glove_para['learning_rate'])
    glove.fit(corpus_model.matrix, epochs=glove_para['no_epochs'],
                  no_threads=glove_para['parallelism'], verbose=True)
    glove.add_dictionary(corpus_model.dictionary)

    glove.save(save_model_name)


def load_default_embed(default_embed_path, words_or_chars):
    """
    example: df_word_emd_load = load_default_embed(default_embed_path=word_embed, 'words')
    :param default_embed_path: word_embed or char_embed
    :param words_or_chars:
    :return: pandas dataframe
    """
    default_embed_df = pd.read_csv(default_embed_path, delim_whitespace=True, index_col=0, header=None)
    cnt_column = default_embed_df.shape[1]
    columns = map(lambda x: "word2vec_w_" + str(x), range(cnt_column))
    default_embed_df.columns = columns
    i2w = {}
    w2i = {}
    for i, w in enumerate(default_embed_df.index.values):
        i2w[i] = w
        w2i[w] = i
    return default_embed_df, i2w, w2i


def eval_glove_single(default_emd, glove_ana, word_str, i2w, w2i, number = 200):
    """
    default_emd: the loaded default embedding model
    glove_ana: the loaded glove embedding model
    word_str: the word to search
    """
    # Evaluation of glove result with provided word embedding
    top_list_glove = glove_ana.most_similar(word_str, number=number)
    top_set_glove = set(map(lambda x: x[0], top_list_glove))

    default_emb_distance = pairwise.pairwise_distances(default_emd)
    top_list_default = set(np.argsort(default_emb_distance[w2i[word_str]], axis=0)[1:number+1])
    top_set_default = set()
    for index in top_list_default:
        top_set_default.add(i2w[index])

    int_set = top_set_glove.intersection(top_set_default)
    uni_set = top_set_glove.union(top_set_default)
    jacc = len(int_set) / len(uni_set)
    return jacc


def eval_glove_full(default_emd, glove_ana, i2w, w2i, number = 200):
    """
    default_emd: the loaded default embedding model
    glove_ana: the loaded glove embedding model
    word_str: the word to search
    """
    # Evaluation of glove result with provided word embedding

    default_emb_distance = pairwise.pairwise_distances(default_emd)
    jacc_full = []

    start = time.time()
    for default_index in xrange(0, default_emd.shape[0]):
        try:
            top_list_default = set(np.argsort(default_emb_distance[default_index], axis=0)[1:number+1])

            word_str = i2w[default_index]
            top_list_glove = glove_ana.most_similar(word_str, number=number)
            top_set_glove = set(map(lambda x: x[0], top_list_glove))

            top_set_default = set()
            for index in top_list_default:
                top_set_default.add(i2w[index])

            int_set = top_set_glove.intersection(top_set_default)
            uni_set = top_set_glove.union(top_set_default)
            jacc = len(int_set) / len(uni_set)
            jacc_full.append(jacc)
        except:
            print default_index
    end = time.time()
    print 'time_spent_on_eval (seconds): ', end - start
    print len(jacc_full)
    print jacc_full[:10]
    jacc_avg = np.average(jacc_full)
    return jacc_avg


def main_train():
    question_path = '../../input/question.csv'
    char_embed = "../../input/char_embed.txt"
    word_embed = "../../input/word_embed.txt"
    question_df = pd.read_csv(question_path)

    # words:
    with open('report_words.txt', 'a+') as report:
        target_group = 'words'
        for learning_rate in [0.001]:
          for no_epochs in [1]:
            glove_para_word = {'window_size': 1, 'no_components': 300, 'learning_rate': learning_rate, 'no_epochs': no_epochs,
                                      'parallelism': 4}
            save_model_name = 'glove_words_0.001_1_win1.model'
            train_glove(target_group=target_group, glove_para=glove_para_word, src_file=question_df, save_model_name=save_model_name)
            df_word_emb_load, i2w, w2i = load_default_embed(default_embed_path=word_embed, words_or_chars=target_group)
            # glove_ana = Glove.load(save_model_name)

            # jacc = eval_glove_single(default_emd=df_word_emb_load, glove_ana= glove_ana, word_str='W00000', i2w=i2w, w2i=w2i, number=3000)
            # jacc_avg = eval_glove_full(default_emd=df_word_emb_load, glove_ana= glove_ana, i2w=i2w, w2i=w2i, number=2000)
            # print jacc_avg
            # report.write(json.dumps(glove_para_word))
            # report.write('jacc_avg: {0} \n'.format(jacc_avg))

    # chars:
    with open('report_chars.txt', 'a+') as report:
        target_group = 'chars'
        for learning_rate in [0.001]:
          for no_epochs in [1]:
            glove_para_word = {'window_size': 3, 'no_components': 300, 'learning_rate': learning_rate, 'no_epochs': no_epochs,
                                      'parallelism': 4}
            save_model_name = 'glove_chars_0.001_1_win3.model'
            train_glove(target_group=target_group, glove_para=glove_para_word, src_file=question_df, save_model_name=save_model_name)
            df_word_emb_load, i2w, w2i = load_default_embed(default_embed_path=char_embed, words_or_chars=target_group)
            # glove_ana = Glove.load('save_model_name')

            # jacc = eval_glove_single(default_emd=df_word_emb_load, glove_ana= glove_ana, word_str='W00000', i2w=i2w, w2i=w2i, number=3000)
            # jacc_avg = eval_glove_full(default_emd=df_word_emb_load, glove_ana= glove_ana, i2w=i2w, w2i=w2i, number=500)
            # print jacc_avg
            # report.write(json.dumps(glove_para_word))
            # report.write('jacc_avg: {0} \n'.format(jacc_avg))


def etl_glove(glove_ana, save_txt_path):
    with open(save_txt_path, 'a+') as dictfile:
      for item in glove_ana.dictionary:
        dictfile.write(item + ' ')
        dictfile.write(' '.join(str(i) for i in glove_ana.word_vectors[glove_ana.dictionary[item]]))
        dictfile.write('\n')

def main():
    """
    load the glove model, and extract, save information as txt.
    :return:
    """
    glove_words_model = Glove.load('glove_words_0.001_1_win1.model')
    glove_chars_model = Glove.load('glove_chars_0.001_1_win3.model')
    save_words_path = 'glove_words_embed.txt'
    save_chars_path = 'glove_chars_embed.txt'
    etl_glove(glove_ana=glove_words_model, save_txt_path=save_words_path)
    etl_glove(glove_ana=glove_chars_model, save_txt_path=save_chars_path)


if __name__ == "__main__":
    main()