# -*- coding: utf-8 -*-
import string
import pandas as pd
import numpy as np
import pickle
import multiprocessing as mp
import gc,os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import issparse
from nltk import ngrams
from sklearn.externals import joblib
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity,\
                                     linear_kernel,\
                                     polynomial_kernel,\
                                     sigmoid_kernel,\
                                     rbf_kernel,\
                                     laplacian_kernel,\
                                     chi2_kernel, \
                                     pairwise_distances



def _do_get_question_embedding_from_term_embedding_task(question_df, term_embed_df, space, w2i, tfidf_mat, avergemethod,
                                                        slice_num, queue):
    print "subprocess %s is doing _do_get_question_embedding_from_term_embedding_task: %s" % (
    os.getpid(), question_df.shape[0])
    res = []
    for i in xrange(question_df.shape[0]):
        cur_row = question_df.iloc[i]
        term_list = cur_row[space].split(" ")
        if avergemethod == "equally":
            res.append(Feature.wemb2semb(term_embed_df, term_list))
        else:
            w_index = [w2i[t] for t in term_list]
            tf_idf_v = np.asarray(tfidf_mat[i, w_index].todense()).ravel()
            if avergemethod == "tf_idf_linear":
                term_weight = tf_idf_v / np.sum(tf_idf_v)
            elif avergemethod == "tf_idf_exp":
                term_weight = F.softmax(tf_idf_v)
            else:
                raise ValueError("Uncognized value (%s) for parameter 'averagemethod'." % (avergemethod))
            res.append(Feature.wemb2semb(term_embed_df, term_list, term_weight))
    queue.put((np.array(res), slice_num))

def _do_get_question_pairs_embedding_comparison_feature_set_task(p1_list,p2_list,fea_func_list,question_embed_df,slice_num,queue):
    print "subprocess %s is doing _do_get_question_pairs_embedding_comparison_feature_set_task: %s" % (
        os.getpid(), len(p1_list))
    res = []
    for i, (p1, p2) in enumerate(zip(p1_list, p2_list)):
        if i%2000==0:
            print os.getpid(),i
        p1_ind = int(p1[1:])
        p2_ind = int(p2[1:])
        p1_emb = question_embed_df[p1_ind:p1_ind + 1]
        p2_emb = question_embed_df[p2_ind:p2_ind + 1]
        res.append([f(p1_emb, p2_emb)[0][0] for f in fea_func_list])
    queue.put((np.array(res),slice_num))

def _do_get_question_pairs_WMD_disctance_task(p1_list,p2_list,question_df,term_embed_df,space,slice_num,queue):
    print "subprocess %s is doing _do_get_question_pairs_WMD_disctance_task: %s" % (os.getpid(), len(p1_list))
    df = []
    i=0
    for p1,p2 in zip(p1_list,p2_list):
        i+=1
        if i%2000==0:
            print os.getpid(),i
        p1_ind = int(p1[1:])
        p2_ind = int(p2[1:])
        p1_terms = question_df.iloc[p1_ind][space].split()
        p2_terms = question_df.iloc[p2_ind][space].split()
        p1_term_embed = term_embed_df.loc[p1_terms].values
        p2_term_embed = term_embed_df.loc[p2_terms].values
        dis_matrix = pairwise_distances(p1_term_embed,p2_term_embed,metric="euclidean")
        res = dis_matrix.min(axis=1).sum()
        df.append([res])
    queue.put((np.array(df),slice_num))

def euclidean(v1, v2):
    return pairwise_distances(v1, v2, metric="euclidean")

def cityblock(v1, v2):
    return pairwise_distances(v1, v2, metric="cityblock")

def my_chi2_kernel(v1,v2):
    if issparse(v1) and issparse(v2):
        v1 = np.asarray(v1.todense())
        v2 = np.asarray(v2.todense())
    return chi2_kernel(v1,v2)

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
    def load_term_embed(cls,space):
        if space=="words":
            return DataSet.load_word_embed()
        else:
            return DataSet.load_char_embed()
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

    @staticmethod
    def load_inters(path):
        if path.endswith(".npz"):
            return sparse.load_npz(path)
        if path.endswith(".npy"):
            return np.load(path)
        if path.endswith("pkl"):
            return pickle.load(open(path,"rb"))
        if path.endswith("csv"):
            return pd.read_csv(path, delim_whitespace=True, index_col=0, header=None)

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
    def chi2_kernel(v1,v2):
        if issparse(v1) and issparse(v2):
            v1 = np.asarray(v1.todense())
            v2 = np.asarray(v2.todense())
        return chi2_kernel(v1,v2)

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
        union = set1.union(set2)
        return len(set1.intersection(set2))/float(len(union))

    @staticmethod
    def if_starts_same(s1,s2):
        if s1[0]==s2[0]:
            return 1
        else:
            return 0

    @staticmethod
    def if_ends_same(s1,s2):
        if s1[-1]==s2[-1]:
            return 1
        else:
            return 0

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
        svder = TruncatedSVD(n_components=num_component,algorithm='arpack')
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
        return svd_U,svd_word_embeddf,svder

    @staticmethod
    def wemb2semb(term_embed_df,term_list,term_weights=None):
        """

        :param term_embed_df: word_embed_dataframe, words as index
        :param term_list: the word list we need to form sentence embedding
        :param term_weights: the weights of all words. default is None, equally weighted
        :return: sentence embedding
        """
        term_embeds = term_embed_df.loc[term_list]
        if term_weights is None:
            return term_embeds.values.mean(axis=0)
        else:
            return (term_embeds.values*(np.array(term_weights).reshape(-1,1))).sum(axis=0)


    @staticmethod
    def get_question_embedding_from_term_embedding(question_df,term_embed_df,space,w2i,tfidf_mat,avergemethod,par=True):
        """
        :param question_df: the question df
        :param term_embed_df: the word or char embed dataframe like word2vec, glove, or result from svd
        :param space: "word" or "char"
        :param w2i: tf_idf vectorizer, optional, used only when 'averagemethod' is "tf_idf"
        :param tfidf_mat: tf_idf matrix, optional, used only when 'averagemethod' is 'tf_idf_linear' or "tf_idf_exp"
        :param avergemethod:"equally","tf_idf_linear","tf_idf_exp"
        :return:question embedding as numpy array, just like the results from tf-idf(sparse) and svd
        """
        if not par:
            res = []
            for i in xrange(question_df.shape[0]):
                if i % 2000 == 0:
                    print i
                cur_row = question_df.iloc[i]
                term_list = cur_row[space].split(" ")
                if avergemethod == "equally":
                    res.append(Feature.wemb2semb(term_embed_df, term_list))
                else:
                    w_index = [w2i[t] for t in term_list]
                    tf_idf_v = np.asarray(tfidf_mat[i, w_index].todense()).ravel()
                    if avergemethod == "tf_idf_linear":
                        term_weight = tf_idf_v / np.sum(tf_idf_v)
                    elif avergemethod == "tf_idf_exp":
                        term_weight = F.softmax(tf_idf_v)
                    else:
                        raise ValueError("Uncognized value (%s) for parameter 'averagemethod'." % (avergemethod))
                    res.append(Feature.wemb2semb(term_embed_df, term_list, term_weight))
            return np.array(res)
        else:
            m = mp.Manager()
            queue = m.Queue()
            p = mp.Pool()
            cpu_cnt = mp.cpu_count()
            ques_cnt = question_df.shape[0]
            cnt_per_cpu = ques_cnt / cpu_cnt if ques_cnt % cpu_cnt == 0 else ques_cnt / cpu_cnt + 1
            for i in xrange(cpu_cnt):
                print i
                if cnt_per_cpu * (i + 1) > ques_cnt:
                    end = ques_cnt
                else:
                    end = cnt_per_cpu * (i + 1)
                p.apply_async(_do_get_question_embedding_from_term_embedding_task,
                              args=(question_df.iloc[cnt_per_cpu * i:end],
                                    term_embed_df,
                                    space,
                                    w2i,
                                    tfidf_mat[cnt_per_cpu * i:end],
                                    avergemethod,
                                    i,
                                    queue))
            print "waiting.."
            p.close()
            p.join()
            tmp_res = []
            while not queue.empty():
                tmp_res.append(queue.get())
            map_id_res = {}
            for res in tmp_res:
                map_id_res[res[1]] = res[0]
            final = []
            for slice in range(cpu_cnt):
                final.append(map_id_res[slice])
            return np.vstack(final)


    @staticmethod
    def get_question_pairs_embedding_comparison_feature_set(p1_list,p2_list,fea_func_list,fea_name_prefix,question_embed_df,par=True):
        """
        Features regarding distance between two vectors, i.e. map (v1,v2) -> scala
        :param p1_list:
        :param p2_list:
        :param fea_func_list:
        :param fea_name_prefix:
        :param question_embed_df: array not dataframe
        :return:
        """
        if not par:
            res = []
            for i,(p1,p2) in enumerate(zip(p1_list,p2_list)):
                if i%2000==0:
                    print i
                p1_ind = int(p1[1:])
                p2_ind = int(p2[1:])
                p1_emb = question_embed_df[p1_ind:p1_ind+1]
                p2_emb = question_embed_df[p2_ind:p2_ind+1]
                res.append([f(p1_emb,p2_emb)[0][0] for f in fea_func_list])

            columns = [fea_name_prefix+"_"+f.__name__ for f in fea_func_list]
            return pd.DataFrame(res,columns=columns)
        else:
            m = mp.Manager()
            queue = m.Queue()
            p = mp.Pool()
            cpu_cnt = mp.cpu_count()
            ques_cnt = len(p1_list)
            cnt_per_cpu = ques_cnt / cpu_cnt if ques_cnt % cpu_cnt == 0 else ques_cnt / cpu_cnt + 1
            for i in xrange(cpu_cnt):
                print i
                if cnt_per_cpu * (i + 1) > ques_cnt:
                    end = ques_cnt
                else:
                    end = cnt_per_cpu * (i + 1)
                p.apply_async(_do_get_question_pairs_embedding_comparison_feature_set_task,
                              args=(p1_list[cnt_per_cpu * i:end],
                                    p2_list[cnt_per_cpu * i:end],
                                    fea_func_list,
                                    question_embed_df,
                                    i,
                                    queue))
            print "waiting.."
            p.close()
            p.join()
            tmp_res = []
            while not queue.empty():
                tmp_res.append(queue.get())
            map_id_res = {}
            for res in tmp_res:
                map_id_res[res[1]] = res[0]
            final = []
            for slice in range(cpu_cnt):
                final.append(map_id_res[slice])
            columns = [fea_name_prefix+"_"+f.__name__ for f in fea_func_list]
            return pd.DataFrame(np.vstack(final),columns=columns)


    @staticmethod
    def get_question_pairs_WMD_disctance(p1_list,p2_list,question_df,term_embed_df,fea_name_prefix,space,par=True):
        if not par:
            df = []
            i=0
            for p1,p2 in zip(p1_list,p2_list):
                i+=1
                if i%2000==0:
                    print i
                p1_ind = int(p1[1:])
                p2_ind = int(p2[1:])
                p1_terms = question_df.iloc[p1_ind][space].split()
                p2_terms = question_df.iloc[p2_ind][space].split()
                p1_term_embed = term_embed_df.loc[p1_terms].values
                p2_term_embed = term_embed_df.loc[p2_terms].values
                dis_matrix = pairwise_distances(p1_term_embed,p2_term_embed,metric="euclidean")
                res = dis_matrix.min(axis=1).sum()
                df.append([res])
            columns = [fea_name_prefix+"_"+space+"_"+"WMD"]
            return pd.DataFrame(df,columns=columns)
        else:
            m = mp.Manager()
            queue = m.Queue()
            p = mp.Pool()
            cpu_cnt = mp.cpu_count()
            ques_cnt = len(p1_list)
            cnt_per_cpu = ques_cnt / cpu_cnt if ques_cnt % cpu_cnt == 0 else ques_cnt / cpu_cnt + 1
            for i in xrange(cpu_cnt):
                print i
                if cnt_per_cpu * (i + 1) > ques_cnt:
                    end = ques_cnt
                else:
                    end = cnt_per_cpu * (i + 1)
                p.apply_async(_do_get_question_pairs_WMD_disctance_task,
                              args=(p1_list[cnt_per_cpu * i:end],
                                    p2_list[cnt_per_cpu * i:end],
                                    question_df,
                                    term_embed_df,
                                    space,
                                    i,
                                    queue))
            print "waiting.."
            p.close()
            p.join()
            tmp_res = []
            while not queue.empty():
                tmp_res.append(queue.get())
            map_id_res = {}
            for res in tmp_res:
                map_id_res[res[1]] = res[0]
            final = []
            for slice in range(cpu_cnt):
                final.append(map_id_res[slice])
            columns = [fea_name_prefix + "_" + "WMD"]
            return pd.DataFrame(np.vstack(final),columns=columns)

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
        i=0
        for p1,p2 in zip(p1_list,p2_list):
            i+=1
            if i%2000==0:
                print i
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
        print len(p1_emb),len(p2_emb)
        if mode=="keep":
            columns = [fea_name_prefix + "_orig_p1_" + str(i) for i in xrange(len(p1_emb))] +\
                      [fea_name_prefix + "_orig_p2_" + str(i) for i in xrange(len(p2_emb))]
        elif mode=="absdif":
            columns = [fea_name_prefix + "_absdif_" + str(i) for i in xrange(len(p1_emb))]
        else:
            columns = [fea_name_prefix + "_orig_p1_" + str(i) for i in xrange(len(p1_emb))] + \
                      [fea_name_prefix + "_orig_p2_" + str(i) for i in xrange(len(p2_emb))] + \
                      [fea_name_prefix + "_absdif_" + str(i) for i in xrange(len(p1_emb))]

        return pd.DataFrame(df,columns=columns)


    @staticmethod
    def get_question_pairs_metadata_feature_set(p1_list,p2_list,fea_func_list,question_df,space):
        df = []
        i = 0
        for p1,p2 in zip(p1_list,p2_list):
            i+=1
            if i%2000==0:
                print i
            p1_ind = int(p1[1:])
            p2_ind = int(p2[1:])
            p1_sen = question_df.iloc[p1_ind][space].split()
            p2_sen = question_df.iloc[p2_ind][space].split()
            try:
                res = np.array([f(p1_sen,p2_sen) for f in fea_func_list])
            except:
                print p1,p1_sen,p2,p2_sen
                exit(0)
            df.append(res)
        columns = [f.__name__+ "_"+ space for f in fea_func_list]
        return pd.DataFrame(df,columns=columns)



def generate_senten_embed_comparion_features(dataset):
    inters_folder = "../../inters/embeddings"
    if dataset=="train":
        data = DataSet.load_train()
    else:
        data = DataSet.load_test()
    p1_list = data["q1"]
    p2_list = data["q2"]
    # sentence_embeddings = ["words_3gram_tfidf_SVD_sentence_embed.npy",
    #                        "words_3gram_tfidf_SVD_word_embed_to_sentence_embed_mode_equally.npy",
    #                        "words_3gram_tfidf_SVD_word_embed_to_sentence_embed_mode_tf_idf_exp.npy",
    #                        "words_3gram_tfidf_SVD_word_embed_to_sentence_embed_mode_tf_idf_linear.npy",
    #                        "word2vec_word_embed_to_sentence_embed_mode_equally.npy",
    #                        "word2vec_word_embed_to_sentence_embed_mode_tf_idf_exp.npy",
    #                        "word2vec_word_embed_to_sentence_embed_mode_tf_idf_linear.npy",
    #                        "chars_5gram_tfidf_SVD_sentence_embed.npy",
    #                        "chars_5gram_tfidf_SVD_char_embed_to_sentence_embed_mode_equally.npy",
    #                        "chars_5gram_tfidf_SVD_char_embed_to_sentence_embed_mode_tf_idf_exp.npy",
    #                        "chars_5gram_tfidf_SVD_char_embed_to_sentence_embed_mode_tf_idf_linear.npy",
    #                        "word2vec_char_embed_to_sentence_embed_mode_equally.npy",
    #                        "word2vec_char_embed_to_sentence_embed_mode_tf_idf_exp.npy",
    #                        "word2vec_char_embed_to_sentence_embed_mode_tf_idf_linear.npy",
    #                        "words_3gram_sentence_tfidf_embedding.npz",
    #                        "chars_5gram_sentence_tfidf_embedding.npz"
    #                        ]
    # sentence_embeddings = [
    #     "glove_char_embed_to_sentence_embed_mode_equally.npy",
    #     "glove_char_embed_to_sentence_embed_mode_tf_idf_exp.npy",
    #     "glove_char_embed_to_sentence_embed_mode_tf_idf_linear.npy",
    #     "glove_word_embed_to_sentence_embed_mode_equally.npy",
    #     "glove_word_embed_to_sentence_embed_mode_tf_idf_exp.npy",
    #     "glove_word_embed_to_sentence_embed_mode_tf_idf_linear.npy",
    # ]

    sentence_embeddings = ["chars_1gram_tfidf_NMF_sentence_embed.npy",
                           "words_1gram_tfidf_NMF_sentence_embed.npy",
                           "chars_1gram_tf_LDA_sentence_embed.npy",
                           "words_1gram_tf_LDA_sentence_embed.npy"]
    feature_list1_for_nottfidf =    [cosine_similarity,
                                    linear_kernel,
                                    polynomial_kernel,
                                    sigmoid_kernel,
                                    rbf_kernel,
                                    laplacian_kernel,
                                    euclidean,
                                    cityblock]
    feature_list2_for_tfidf =   [cosine_similarity,
                                polynomial_kernel,
                                sigmoid_kernel,
                                rbf_kernel,
                                laplacian_kernel,
                                my_chi2_kernel,
                                euclidean,
                                cityblock]

    for i,sepath in enumerate(sentence_embeddings):

        print i,"doing", dataset, sepath

        fea_func_list = feature_list1_for_nottfidf
        fea_name_prefix = sepath.split(".")[0]
        question_embed_df = DataSet.load_inters(os.path.join(inters_folder,sepath))
        partial_df = Feature.get_question_pairs_embedding_comparison_feature_set(p1_list,
                                                                                 p2_list,
                                                                                 fea_func_list,
                                                                                 fea_name_prefix,
                                                                                 question_embed_df,
                                                                                 par=True)
        senten_embed_comparion_outfolder = "../../inters/partial_features"
        senten_embed_comparion_file = dataset+"_"+fea_name_prefix+"_comp_feature.csv"
        partial_df.to_csv(os.path.join(senten_embed_comparion_outfolder,
                                       senten_embed_comparion_file),index=False)

def generate_question_pairs_WMD_distance_features(dataset):
    """
    p1_list,p2_list,question_df,term_embed_df,fea_name_prefix,space,par=True
    :param dataset:
    :return:
    """
    inters_folder = "../../inters/embeddings"
    if dataset=="train":
        data = DataSet.load_train()
    else:
        data = DataSet.load_test()
    p1_list = data["q1"]
    p2_list = data["q2"]
    question_df = DataSet.load_all_questions()
    term_embed_df = ["words_3gram_tfidf_SVD_word_embed.csv",
                     "word2vec_word_embed.csv",
                     "glove_word_embed.csv",
                     "chars_5gram_tfidf_SVD_char_embed.csv",
                     "word2vec_char_embed.csv",
                     "glove_char_embed.csv"
                     ]
    spaces = ["words","words","words","chars","chars","chars"]
    for tb,space in zip(term_embed_df,spaces):
        term_embed = DataSet.load_inters(os.path.join(inters_folder,tb))
        fea_name_prefix = tb.split(".")[0]
        fea_df = Feature.get_question_pairs_WMD_disctance(p1_list,p2_list,question_df,term_embed,fea_name_prefix,space,True)
        WMD_outfolder = "../../inters/partial_features"
        WMD_file = dataset+"_"+fea_name_prefix+"_WMD_feature.csv"
        fea_df.to_csv(os.path.join(WMD_outfolder,WMD_file),index=False)

def generate_question_pairs_meta_features(dataset):
    """
    get_question_pairs_metadata_feature_set(p1_list,p2_list,fea_func_list,question_df,space)
    :param dataset:
    :return:
    """
    if dataset=="train":
        data = DataSet.load_train()
    else:
        data = DataSet.load_test()
    p1_list = data["q1"]
    p2_list = data["q2"]
    question_df = DataSet.load_all_questions()
    spaces = ["words","chars"]
    n = 3
    fea_func_list =              [F.spair_len_s1,\
                                  F.spair_len_s2,\
                                  F.spair_len_dif_abs,\
                                  F.spair_len_dif_over_max,\
                                  F.spair_len_dif_over_min,\
                                  F.spair_len_dif_over_mean,\
                                  F.levenshtein,\
                                  F.if_starts_same,\
                                  F.if_ends_same
                                  ]+\
                                 [F.num_commom_n_gram(i) for i in range(1,n+1)]+\
                                 [F.jaccard_till_n_gram(n)]
    for space in spaces:
        fea_df = Feature.get_question_pairs_metadata_feature_set(p1_list,p2_list,fea_func_list,question_df,space)
        outfolder = "../../inters/partial_features"
        file = dataset+"_"+space+"_meta_feature.csv"
        fea_df.to_csv(os.path.join(outfolder,file),index=False)


def generate_question_pairs_embed_orig_features(dataset):
    """
    get_question_pairs_embedding_original_feature_set(p1_list,p2_list,fea_name_prefix,question_embed_df,mode)
    :param dataset:
    :return:
    """
    inters_folder = "../../inters/embeddings"
    if dataset=="train":
        data = DataSet.load_train()
    else:
        data = DataSet.load_test()
    p1_list = data["q1"]
    p2_list = data["q2"]
    # question_embeds = ["word2vec_word_embed_to_sentence_embed_mode_equally.npy",
    #                   "word2vec_char_embed_to_sentence_embed_mode_equally.npy"]
    question_embeds = ["words_3gram_tfidf_SVD_sentence_embed.npy",
                       "chars_5gram_tfidf_SVD_sentence_embed.npy",
                       "glove_char_embed_to_sentence_embed_mode_tf_idf_linear.npy",
                       "glove_word_embed_to_sentence_embed_mode_tf_idf_linear.npy",
                       "words_1gram_tf_LDA_sentence_embed.npy",
                       "chars_1gram_tf_LDA_sentence_embed.npy",
                       "chars_1gram_tfidf_NMF_sentence_embed.npy",
                       "words_1gram_tfidf_NMF_sentence_embed.npy"
                       ]
    for qe_path in question_embeds:
        prefix = qe_path.split(".")[0]
        ques_emb = DataSet.load_inters(os.path.join(inters_folder,qe_path))
        res = Feature.get_question_pairs_embedding_original_feature_set(p1_list,p2_list,prefix,ques_emb,"absdif")
        outfolder = "../../inters/partial_features"
        outfile = dataset+"_"+prefix+"_orig_feature.csv"
        res.to_csv(os.path.join(outfolder,outfile),index=False)

def gen_and_store_intermediate_resutls_words():
    all_questions = DataSet.load_all_questions()
    spaces = ["words"]
    inters = "../../inters/embeddings"
    for space in spaces:
        n = 3 if space=="words" else 5
        print "loading...."
        words_3gram_vectorizer = DataSet.load_inters(os.path.join(inters,"words_3gram_vectorizer.pkl"))
        ques_tfidf_embedding_array = DataSet.load_inters(os.path.join(inters,"words_3gram_sentence_tfidf_embedding.npz"))

        gove_embedding = DataSet.load_inters(os.path.join(inters,"glove_word_embed.csv"))

        for i, term_embedding in enumerate([gove_embedding]):
            for mode in ["equally", "tf_idf_linear", "tf_idf_exp"]:
                question_emb_from_word_embed = Feature.get_question_embedding_from_term_embedding(all_questions,
                                                                                                  term_embedding,
                                                                                                  space,
                                                                                                  words_3gram_vectorizer.vocabulary_,
                                                                                                  ques_tfidf_embedding_array,
                                                                                                  mode)
                print i, mode

                name = "glove_word_embed_to_sentence_embed"
                np.save(name+"_mode_"+mode,question_emb_from_word_embed)

def gen_and_store_intermediate_resutls_chars():
    all_questions = DataSet.load_all_questions()
    spaces = ["chars"]
    inters = "../../inters/embeddings"
    for space in spaces:
        n = 3 if space=="words" else 5
        print "loading...."
        chars_5gram_vectorizer = DataSet.load_inters(os.path.join(inters,"chars_5gram_vectorizer.pkl"))
        ques_tfidf_embedding_array = DataSet.load_inters(os.path.join(inters,"chars_5gram_sentence_tfidf_embedding.npz"))

        gove_embedding = DataSet.load_inters(os.path.join(inters,"glove_char_embed.csv"))

        for i, term_embedding in enumerate([gove_embedding]):
            for mode in ["equally", "tf_idf_linear", "tf_idf_exp"]:
                question_emb_from_word_embed = Feature.get_question_embedding_from_term_embedding(all_questions,
                                                                                                  term_embedding,
                                                                                                  space,
                                                                                                  chars_5gram_vectorizer.vocabulary_,
                                                                                                  ques_tfidf_embedding_array,
                                                                                                  mode)
                print i, mode

                name = "glove_char_embed_to_sentence_embed"
                np.save(name+"_mode_"+mode,question_emb_from_word_embed)

def combine_feature_pdfs():
    in_folder = "../../inters/partial_features/"
    out_folder = "../../inters/final_features"
    all_feature_files = os.listdir(in_folder)
    all_train_feature_files = [file for file in all_feature_files if file.startswith("train")]
    all_train_features_df = [pd.read_csv(os.path.join(in_folder,file)) for file in all_train_feature_files ]
    train_feature = pd.concat(all_train_features_df,axis=1)
    del all_train_features_df
    gc.collect()
    label = DataSet.load_train()["label"]
    train_feature["label"]=label
    train_feature.to_csv(os.path.join(out_folder,"train_type2.csv"),index=False)
    del train_feature
    gc.collect()
    all_test_feature_files = [file.replace("train","test") for file in all_train_feature_files]
    all_test_features_df = [pd.read_csv(os.path.join(in_folder,file)) for file in all_test_feature_files ]
    test_feature = pd.concat(all_test_features_df,axis=1)
    del all_test_features_df
    gc.collect()
    test_feature.to_csv(os.path.join(out_folder,"test_type2.csv"),index=False)

if __name__ == "__main__":
    #gen_and_store_intermediate_resutls_chars()
    # qs = DataSet.load_all_questions()
    # we = DataSet.load_word_embed()
    # v = pickle.load(open("../../inters/embeddings/words_3gram_vectorizer.pkl", "rb"))
    # w2i = v.vocabulary_
    # tfidf = sparse.load_npz("../../inters/embeddings/words_3gram_sentence_tfidf_embedding.npz")
    # q_emb = Feature.get_question_embedding_from_term_embedding(qs, we, "words", w2i, tfidf, "equally", True)
    # np.save("word2vec_word_to_sentence_equally_test",q_emb)
    # train = DataSet.load_train()
    # p1 = train["q1"][:1000]
    # p2 = train["q2"][:1000]
    # compfeature_list = [cosine_similarity, \
    #                     linear_kernel, \
    #                     polynomial_kernel, \
    #                     sigmoid_kernel, \
    #                     rbf_kernel, \
    #                     laplacian_kernel,\
    #                     euclidean,\
    #                     cityblock]
    # embedding = np.load("word2vec_word_to_sentence_equally_test.npy")
    #embedding = sparse.load_npz("../../inters/embeddings/chars_5gram_sentence_tfidf_embedding.npz")
    #print embedding.shape
    #fea_name_prefix = "original"
    #onethread = Feature.get_question_pairs_embedding_comparison_feature_set(p1,p2,compfeature_list,prefix,embedding,False)
    #onethread.to_csv("chars_5gram_sentence_tfidf_embedding_one_thread.csv",index=False)
    #multiple = Feature.get_question_pairs_embedding_comparison_feature_set(p1,p2,compfeature_list,prefix,embedding,True)
    #multiple.to_csv("chars_5gram_sentence_tfidf_embedding_multi_thread.csv",index=False)
    #space = "words"
    #term_embed = DataSet.load_term_embed(space)
    #qs = DataSet.load_all_questions()
    #WMD = Feature.get_question_pairs_embedding_original_feature_set(p1,p2,fea_name_prefix,embedding,"keep|absdif")
    #WMD.to_csv("origtest.csv")
    #WMD_false = Feature.get_question_pairs_WMD_disctance(p1,p2,qs,term_embed,"",space,par=False)
    #WMD_false.to_csv("WMDfalse.csv",index=False)
    #WMD_true = Feature.get_question_pairs_WMD_disctance(p1,p2,qs,term_embed,"",space,par=True)
    #WMD_true.to_csv("WMDteur.csv",index=False)
    #n=3
    #meta_feature_list = [F.spair_len_s1,\
                                #  F.spair_len_s2,\
                                #  F.spair_len_dif_abs,\
                                #  F.spair_len_dif_over_max,\
                                #  F.spair_len_dif_over_min,\
                                #  F.spair_len_dif_over_mean,\
                                #  F.levenshtein,\
                                #  ]+\
                                # [F.num_commom_n_gram(i) for i in range(1,n+1)]+\
                                # [F.jaccard_till_n_gram(n)]

    #meta_test=  Feature.get_question_pairs_metadata_feature_set(p1, p2, meta_feature_list, qs, "words")
    #meta_test.to_csv("metatest.csv",index=False)
    # generate_senten_embed_comparion_features("train")
    # generate_senten_embed_comparion_features("test")
    #generate_question_pairs_WMD_distance_features("train")
    #generate_question_pairs_WMD_distance_features("test")
    # generate_question_pairs_meta_features("train")
    # generate_question_pairs_meta_features("test")
    # generate_question_pairs_embed_orig_features("train")
    # generate_question_pairs_embed_orig_features("test")
    #gen_and_store_intermediate_resutls_words()
    #gen_and_store_intermediate_resutls_chars()
    #combine_feature_pdfs()
    combine_feature_pdfs()

