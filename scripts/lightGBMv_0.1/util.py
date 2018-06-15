# -*- coding: utf-8 -*-
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import issparse
from nltk import ngrams
from sklearn.metrics.pairwise import cosine_similarity,\
                                     linear_kernel,\
                                     polynomial_kernel,\
                                     sigmoid_kernel,\
                                     rbf_kernel,\
                                     laplacian_kernel,\
                                     chi2_kernel, \
                                     pairwise_distances

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
    def load_all_questions(cls):
        question_df = pd.read_csv(cls.question_path,index_col=0)
        return question_df
    @classmethod
    def load_necessary_questions(cls):
        """
        Load only necessary questions, that is, all questions appear in train and test question pairs
        :return:
        """
        return DataSet.load_all_questions().loc[DataSet.load_all_unique_ids_train_test()]
    @classmethod
    def load_corpus(cls,space):
        """
        Load corpus, i.e. all questions. Space indicates "words" or "chars"
        The corpus can be used to train our own word or char embedding and idfs
        :param space:
        :return:
        """
        return DataSet.load_all_questions()[space]
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
    @classmethod
    def load_unique_train_qids(cls):
        train_df = DataSet.load_train()
        all_qs = train_df["q1"].values.tolist() + train_df["q2"].values.tolist()
        return list(set(all_qs))
    @classmethod
    def load_unique_test_qids(cls):
        test_df = DataSet.load_test()
        all_qs = test_df["q1"].values.tolist() + test_df["q2"].values.tolist()
        return list(set(all_qs))
    @classmethod
    def load_all_unique_ids_train_test(cls):
        return list(set(DataSet.load_unique_train_qids()+DataSet.load_unique_test_qids()))



class F(object):
    @staticmethod
    def reverse_dictionary(input_dict):
        """reverse the key-value pair into value-key pair"""
        return {v:k for k,v in input_dict.items()}

    @staticmethod
    def get_n_gram(s,n):
        grams = ngrams(s, n)
        grams = [' '.join(x) for x in grams]
        return grams

    @staticmethod
    def levenshtein(s1, s2):
        if len(s1) < len(s2):
            return F.levenshtein(s2, s1)

        # len(s1) >= len(s2)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
                deletions = current_row[j] + 1  # than s2
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    @staticmethod
    def softmax(arr):
        e_x = np.exp(arr-np.max(arr))
        return e_x/np.sum(e_x)

    @staticmethod
    def euclidean(v1,v2):
        return pairwise_distances(v1,v2,metric="euclidean")

    @staticmethod
    def cityblock(v1,v2):
        return pairwise_distances(v1,v2,metric="cityblock")

    @staticmethod
    def spair_len_s1(s1,s2):
        return len(s1)

    @staticmethod
    def spair_len_s2(s1,s2):
        return len(s2)

    @staticmethod
    def spair_len_dif_abs(s1,s2):
        return abs(len(s1)-len(s2))

    @staticmethod
    def spair_len_dif_over_max(s1,s2):
        return F.spair_len_dif_abs(s1,s2)/max(len(s1),len(s2))

    @staticmethod
    def spair_len_dif_over_min(s1,s2):
        return F.spair_len_dif_abs(s1, s2) / min(len(s1), len(s2))

    @staticmethod
    def spair_len_dif_over_mean(s1,s2):
        return F.spair_len_dif_abs(s1,s2)*2/(len(s1)+len(s2))

    @staticmethod
    def jaccard_disance(set1,set2):
        return len(set1.intersection(set2))/float(len(set1.union(set2)))

    @staticmethod
    def levenshtein_sentence(s1,s2):
        term_set = list(set(s1+s2))
        char_set = string.printable
        mapping = {}
        for i,t in enumerate(term_set):
            mapping[t]=char_set[i]
        s1_str = ''.join([mapping[t] for t in s1])
        s2_str = ''.join([mapping[t] for t in s2])
        return F.levenshtein(s1_str,s2_str)


    class num_commom_n_gram(object):
        def __init__(self,n):
            self.n = n
            self.__name__ = "num_commom_%s_gram"%(n)
        def __call__(self,s1,s2):
            """

            :param s1: list of terms
            :param s2: list of terms
            :return:
            """
            s1_gram = F.get_n_gram(s1,self.n)
            s2_gram = F.get_n_gram(s2,self.n)
            return len(set(s1_gram).intersection(set(s2_gram)))

    class num_common_till_n_gram(object):
        def __init__(self,n):
            self.n = n
            self.__name__ = "num_commom_till_%s_gram"%(n)
        def __call__(self,s1,s2):
            s1_gram = reduce(lambda x,y:x+y,[F.get_n_gram(s1,i) for i in range(1,self.n+1)])
            s2_gram = reduce(lambda x,y:x+y,[F.get_n_gram(s2,i) for i in range(1,self.n+1)])
            return len(set(s1_gram).intersection(set(s2_gram)))

    class jaccard_n_gram(object):
        def __init__(self,n):
            self.n = n
            self.__name__ = "jaccard_%s_gram"%(n)
        def __call__(self,s1,s2):
            s1_gram = F.get_n_gram(s1,self.n)
            s2_gram = F.get_n_gram(s2,self.n)
            return F.jaccard_disance(set(s1_gram),set(s2_gram))

    class jaccard_till_n_gram(object):
        def __init__(self,n):
            self.n = n
            self.__name__ = "jaccard_till_%s_gram"%(n)
        def __call__(self,s1,s2):
            s1_gram = reduce(lambda x,y:x+y,[F.get_n_gram(s1,i) for i in range(1,self.n+1)])
            s2_gram = reduce(lambda x,y:x+y,[F.get_n_gram(s2,i) for i in range(1,self.n+1)])
            return F.jaccard_disance(set(s1_gram),set(s2_gram))



class Feature(object):
    @staticmethod
    def get_ques_term_tfidf_matrix(question_df,space,max_n_gram):
        vectorizer = TfidfVectorizer(ngram_range=(1,max_n_gram),lowercase=False)
        corpus = DataSet.load_corpus(space)
        vectorizer.fit(corpus)
        tfidf_mat = vectorizer.transform(question_df[space])
        return vectorizer,tfidf_mat

    @staticmethod
    def get_SVD_feature(vectorizer,tfidf_mat,num_component):
        svder = TruncatedSVD(n_components=num_component, algorithm='arpack')
        svd_U = svder.fit_transform(tfidf_mat)
        svd_V = svder.components_
        w2i = vectorizer.vocabulary_
        valid_w = []
        valid_i = []
        for w,i in w2i.items():
            if " " not in w: ##find all unigrams, we only want the embedding of unigram terms
                valid_w.append(w)
                valid_i.append(i)
        svd_V = svd_V[:,valid_i]
        svd_word_embeddf = pd.DataFrame(svd_V.T,index=valid_w)
        return svd_U,svd_word_embeddf

    @staticmethod
    def wemb2semb(term_embed_df,term_list,term_weights=None):
        """

        :param term_embed_df: word_embed_dataframe, words as index
        :param term_list: the word list we need to form sentence embedding
        :param term_weights: the weights of all words. default is None, equally weighted
        :return: sentence embedding
        """
        term_embeds = term_embed_df.loc[term_list]
        if term_weights==None:
            return term_embeds.values.mean(axis=0)
        else:
            return (term_embeds.values*(np.array(term_weights).reshape(-1,1))).sum(axis=0)

    @staticmethod
    def get_question_embedding_from_term_embedding(question_df,term_embed_df,space,vectorizer,tfidf_mat,avergemethod):
        """
        :param question_df: the question df
        :param term_embed_df: the word or char embed dataframe like word2vec, glove, or result from svd
        :param space: "word" or "char"
        :param vectorizer: tf_idf vectorizer, optional, used only when 'averagemethod' is "tf_idf"
        :param tfidf_mat: tf_idf matrix, optional, used only when 'averagemethod' is 'tf_idf_linear' or "tf_idf_exp"
        :param avergemethod:"equally","tf_idf_linear","tf_idf_exp"
        :return:question embedding as numpy array, just like the results from tf-idf(sparse) and svd
        """
        res = []
        for i in xrange(question_df.shape[0]):
            cur_row = question_df.iloc[i]
            term_list = cur_row[space].split(" ")
            if avergemethod=="equally":
                res.append(Feature.wemb2semb(term_embed_df,term_list))
            else:
                w_index = [vectorizer.vocabulary_[t] for t in term_list]
                tf_idf_v = np.asarray(tfidf_mat[i,w_index].todense()).ravel()
                if avergemethod=="tf_idf_linear":
                    term_weight = tf_idf_v/np.sum(tf_idf_v)
                elif avergemethod=="tf_idf_exp":
                    term_weight = F.softmax(tf_idf_v)
                else:
                    raise ValueError("Uncognized value (%s) for parameter 'averagemethod'."%(avergemethod))
                res.append(Feature.wemb2semb(term_embed_df,term_list,term_weight))
        return np.array(res)

    @staticmethod
    def get_v2v_feature_set(v1,v2,v2v_funcs):
        """

        :param v1: one sample vector, always 2-dimentional, reshape(1,-1) before calling this function
        :param v2:
        :param v2v_funcs: list of functions to calculate the distance between vectors
        :return:
        """
        v2v_feas = []
        for f in v2v_funcs:
            v2v_feas.append(f(v1,v2)[0][0])
        return np.array(v2v_feas)

    @staticmethod
    def get_question_pairs_embedding_comparison_feature_set(p1_list,p2_list,fea_func_list,fea_name_prefix,question_embed_df):
        """
        Features regarding distance between two vectors, i.e. map (v1,v2) -> scala
        :param p1_list:
        :param p2_list:
        :param fea_func_list:
        :param fea_name_prefix:
        :param question_embed_df: array not dataframe
        :return:
        """
        res = []
        for p1,p2 in zip(p1_list,p2_list):
            p1_ind = int(p1[1:])
            p2_ind = int(p2[1:])
            p1_emb = question_embed_df[p1_ind:p1_ind+1]
            p2_emb = question_embed_df[p2_ind:p2_ind+1]
            res.append(Feature.get_v2v_feature_set(p1_emb,p2_emb,fea_func_list))
        columns = [fea_name_prefix+"_"+f.__name__ for f in fea_func_list]
        return pd.DataFrame(res,columns=columns)

    @staticmethod
    def get_question_pairs_embedding_original_feature_set(p1_list,p2_list,fea_name_prefix,question_embed_df,mode):
        """
        Features mapping (v1,v2) -> vector
        :param p1_list:
        :param p2_list:
        :param fea_name_prefix:
        :param question_embed_df:
        :param mode:
        :return:
        """
        df = []
        for p1,p2 in zip(p1_list,p2_list):
            p1_ind = int(p1[1:])
            p2_ind = int(p2[1:])
            if issparse(question_embed_df):
                p1_emb = np.asarray(question_embed_df[p1_ind].todense()).ravel()
                p2_emb = np.asarray(question_embed_df[p2_ind].todense()).ravel()
            else:
                p1_emb = question_embed_df[p1_ind]
                p2_emb = question_embed_df[p2_ind]
            if mode=="keep":
                res = np.concatenate([p1_emb,p2_emb])
            elif mode=="absdif":
                res = np.abs(p1_emb-p2_emb)
            elif mode=="keep|absdif":
                res = np.concatenate([p1_emb,p2_emb,np.abs(p1_emb-p2_emb)])
            else:
                raise ValueError("Unrecognized value '%s' for parameter 'mode'."%(mode))
            df.append(res)
        if mode=="keep":
            columns = [fea_name_prefix + "_orig_p1_" + str(i) for i in xrange(len(p1_emb))] +\
                      [fea_name_prefix + "_orig_p2_" + str(i) for i in xrange(len(p2_emb))]
        elif mode=="absdif":
            columns = [fea_name_prefix + "_absdif_" + str(i) for i in xrange(len(res))]
        else:
            columns = [fea_name_prefix + "_orig_p1_" + str(i) for i in xrange(len(p1_emb))] + \
                      [fea_name_prefix + "_orig_p2_" + str(i) for i in xrange(len(p2_emb))] + \
                      [fea_name_prefix + "_absdif_" + str(i) for i in xrange(len(res))]

        return pd.DataFrame(df,columns=columns)

    @staticmethod
    def get_s2s_meta_feature_set(s1,s2,func_list):
        return np.array([f(s1,s2) for f in func_list])

    @staticmethod
    def get_question_pairs_metadata_feature_set(p1_list,p2_list,fea_func_list,question_df,space):
        df = []
        for p1,p2 in zip(p1_list,p2_list):
            p1_ind = int(p1[1:])
            p2_ind = int(p2[1:])
            p1_sen = question_df.iloc[p1_ind][space].split()
            p2_sen = question_df.iloc[p2_ind][space].split()


    @staticmethod
    def generate_feature(p1_list,p2_list):
        pass

if __name__ == "__main__":
    #print DataSet.load_word_embed()
    pass