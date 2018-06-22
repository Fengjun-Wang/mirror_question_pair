# -*- coding: utf-8 -*-
import sys
import os
#import pickle
import hashlib
sys.path.append("../")
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
try:
    import cPickle as pickle
except:
    import pickle
from past.builtins import xrange
from feature_engineering.util import DataSet
_PAD_ = "_PAD_"
class GreedyBucket(object):
    def fit(self,list_pair_len):
        self.list_pair_len_max = list(map(max, list_pair_len))
        max_and_pair_list = list(zip(self.list_pair_len_max,list_pair_len))
        max_cost_list = list(map(lambda x:(x[0],2*x[0]-x[1][0]-x[1][1]),max_and_pair_list))
        max_accumu = {}
        for max_,cost_ in max_cost_list:
            tmp_cost_num = max_accumu.setdefault(max_,[0,0])
            tmp_cost_num[0] += cost_
            tmp_cost_num[1] += 1
            max_accumu[max_] = tmp_cost_num
        #print max_accumu
        max_cost_list = [(k,v[0],v[1]) for k,v in max_accumu.items()]## max_len, cost, num
        max_cost_list.sort(key=lambda x:x[0])
        return max_cost_list

    def _binary_search_upbound(self,sort_max_cost_list,split_point):
        i = 0
        j = len(sort_max_cost_list)-1
        while i<=j:
            if sort_max_cost_list[(i+j)/2]>split_point:
                j -= 1
            elif sort_max_cost_list[(i+j)/2]==split_point:
                return (i+j)/2, sort_max_cost_list[(i+j)/2]
            else:
                i+=1
        return i,sort_max_cost_list[i]

    def _linear_search_upbound(self,sort_max_cost_list,split_point):
        for i,v in enumerate(sort_max_cost_list):
            if v>=split_point:
                return i,v

    def _find_split_points(self,sort_max_cost_list,num_bucket):
        num_splits = num_bucket - 1
        if num_splits == 0: ## no need to split, then one bucket, append to the max length
            total_cost = 0
            append_to = sort_max_cost_list[-1][0]
            for k,cost,num in sort_max_cost_list:
                total_cost += 2*(append_to-k)*num+cost
            return [append_to],total_cost

        else:
            cost1 = sort_max_cost_list[0][1] ## first item cost if is is a standalone bucket
            cost2 = 0
            for i in range(1,len(sort_max_cost_list)):
                cost2 += sort_max_cost_list[i][1]+2*(sort_max_cost_list[-1][0]-sort_max_cost_list[i][0])*sort_max_cost_list[i][2]
            total_cost = cost1+cost2 ##cost if splitting on first item
            prefer_index = 0
            accu_num_sum = sort_max_cost_list[0][2]
            for i in range(1,len(sort_max_cost_list)-1): ## iterate splitting on other items, and compare with the first item and decide best splitting point
                cost1 = cost1+2*(sort_max_cost_list[i][0]-sort_max_cost_list[i-1][0])*accu_num_sum ##+sort_max_cost_list[i][1]
                cost2 = cost2-2*(sort_max_cost_list[-1][0]-sort_max_cost_list[i][0])*sort_max_cost_list[i][2]##-sort_max_cost_list[i][1]
                accu_num_sum += sort_max_cost_list[i][2]
                new_total = cost1+cost2
                if new_total < total_cost:
                    total_cost = new_total
                    prefer_index = i
            num_splits -= 1
            next_len_1 = prefer_index
            next_len_2 = len(sort_max_cost_list)-1-next_len_1-1
            bigger = sys.maxsize
            for i in range(num_splits+1):
                split1 = i
                split2 = num_splits-i
                if split1<=next_len_1 and split2<=next_len_2:
                    res1 = self._find_split_points(sort_max_cost_list[:prefer_index+1],split1+1)
                    res2 = self._find_split_points(sort_max_cost_list[prefer_index+1:],split2+1)
                    combine_cost = res1[1]+res2[1]
                    if combine_cost<bigger:
                        bigger = combine_cost
                        splits = res1[0]+res2[0]
            return splits,bigger

    def get_split_results(self,sort_max_cost_list,num_bucket):
        if num_bucket<1 or num_bucket>len(sort_max_cost_list):
            raise ValueError("'bucket_num' must be between 1 and len(sort_max_cost_list)[%s]"%(len(sort_max_cost_list)))
        split_res =  self._find_split_points(sort_max_cost_list,num_bucket)
        split_points = split_res[0]
        split_points.sort()
        bounds = []
        buckets = {}
        for i,max_i in enumerate(self.list_pair_len_max):
            bucket,upbound = self._linear_search_upbound(split_points,max_i)
            buckets.setdefault(bucket,[]).append(i)
            bounds.append(upbound)
        #print split_res
        return buckets,bounds

class PaddedTensorDataset(Dataset):
    def __init__(self, *data_tensors):
        #self.data_tensors = [torch.LongTensor(data) for data in data_tensors]
        self.data_tensors = data_tensors
        lens = np.array([len(data) for data in data_tensors])
        assert not np.any(lens-lens[0])

    def __getitem__(self, index):
        return tuple([datatensor[index] for datatensor in self.data_tensors])

    def __len__(self):
        return len(self.data_tensors[0])

class PaddedTensorDatasetTest(Dataset):
    def __init__(self, data,ind):
        #self.data_tensors = [torch.LongTensor(data) for data in data_tensors]
        self.data_tensor = data
        self.ind = ind
        assert len(data)==len(ind)

    def __getitem__(self, index):
        return self.data_tensor[index],self.ind[index]

    def __len__(self):
        return len(self.data_tensor)

class PaddedTensorDatasetNoTest(Dataset):
    def __init__(self, data,ind,label):
        #self.data_tensors = [torch.LongTensor(data) for data in data_tensors]
        self.data_tensor = data
        self.ind = ind
        self.label = label
        assert len(data)==len(ind)==len(label)

    def __getitem__(self, index):
        return self.data_tensor[index],self.ind[index],self.label[index]

    def __len__(self):
        return len(self.data_tensor)


class DataGenerator(object):
    def __init__(self,name):
        self.generator_name = name
        if os.path.exists(self._temp_file):
            print("detect cached intermediate files...loading...")
            all_cached = pickle.load(open(self._temp_file,"rb"))
            self.item2idx = all_cached["item2idx"]
            self.idx2item = all_cached["idx2item"]
            self.item_embed = all_cached["item_embed"]
            self.q2idvec = all_cached["q2idvec"]
            print("finish")
        else:
            print("Generating intermediate files...")
            self.item2idx = {}
            self.idx2item = {}
            self.item_embed = {}
            self.q2idvec = {}
            spaces = ["words","chars"]
            question_df = DataSet.load_all_questions()
            all_qids = DataSet.load_all_unique_ids_train_test()
            for space in spaces:
                print("for",space)
                corpus = question_df[space]
                w2i,i2w = self._get_item2id_id2item(corpus)
                self.item2idx[space] = w2i
                self.idx2item[space] = i2w
                ##Finish mapping table
                term_embed = DataSet.load_term_embed(space)
                embed_size = term_embed.shape[1]
                pad_embed = np.array([0] * embed_size).reshape(1, -1)
                all_embeding = np.vstack([pad_embed, term_embed])
                all_index = [_PAD_] + term_embed.index.values.tolist()
                all_embeding_df = pd.DataFrame(data=all_embeding, index=all_index)
                sort_word = [i2w[i] for i in range(len(i2w))]
                self.item_embed[space] = all_embeding_df.loc[sort_word].values
                ##Finish item embedding
                tmp_q2idvec = {}
                for qid in all_qids:
                    items = question_df.loc[qid][space].split()
                    idvec = np.array([w2i[w] for w in items])
                    tmp_q2idvec[qid] = idvec
                self.q2idvec[space]=tmp_q2idvec
                ##Finish map from question to id vector
            print("finish generating inter files.")
            print("begin caching..")
            all_cached = {}
            all_cached["item2idx"] =  self.item2idx
            all_cached["idx2item"] =  self.idx2item
            all_cached["item_embed"] = self.item_embed
            all_cached["q2idvec"] = self.q2idvec
            try:
                os.makedirs("./temp")
            except:
                pass
            pickle.dump(all_cached,open(self._temp_file,"wb"))
            print("finish caching")

    def prepare(self,data_df,bucket_num,space,is_prefix_pad,is_test):
        print("prepare necessary data...")
        assert space in ["words","chars"]
        self.is_test = is_test
        self.data_df = data_df
        q_len_title = ["%s_len_q%s"%(space[:-1],i) for i in [1,2]]
        q_pair_list = list(zip(data_df[q_len_title[0]],
                               data_df[q_len_title[1]]
                               )
                           )
        bucket = GreedyBucket()
        fit_res = bucket.fit(q_pair_list)
        buckets,bounds = bucket.get_split_results(fit_res,bucket_num)
        self.buckets= buckets
        self.bounds = bounds
        data_set_id_vectors = []
        q2idvs = self.q2idvec[space]
        print(len(self.data_df))
        for ind in xrange(len(self.data_df)):
            cur_row = data_df.iloc[ind]
            cur_q1 = cur_row["q1"]
            cur_q2 = cur_row["q2"]
            q1_idv = q2idvs[cur_q1]
            q2_idv = q2idvs[cur_q2]
            cur_bound = bounds[ind]
            q1_pad_len = cur_bound - len(q1_idv)
            q2_pad_len = cur_bound - len(q2_idv)
            if is_prefix_pad:
                cur_q1_padded = np.pad(q1_idv,(q1_pad_len,0),"constant")
                cur_q2_padded = np.pad(q2_idv,(q2_pad_len,0),"constant")
            else:
                cur_q1_padded = np.pad(q1_idv,(0,q1_pad_len),"constant")
                cur_q2_padded = np.pad(q2_idv,(0,q2_pad_len),"constant")
            cur_pair_padded = np.concatenate([cur_q1_padded,cur_q2_padded])
            data_set_id_vectors.append(cur_pair_padded)
        self.data_set_id_vectors = np.array(data_set_id_vectors)
        print("prepare done.")

    def get_data_generator(self,is_shuffle,batch_size):
        bucketkeys = list(self.buckets.keys())
        if is_shuffle:
            np.random.shuffle(bucketkeys)
        else:
            bucketkeys.sort() ##if not shuffle, always starts with the bucket with smallest length
        for b in bucketkeys:
            id_list = self.buckets[b]
            tmpdata = np.vstack(self.data_set_id_vectors[id_list])
            if not self.is_test:
                tmplabel = self.data_df["label"].iloc[id_list].values
            tmpids = np.array(id_list)
            tmpsels = np.arange(len(tmpids))
            if is_shuffle:
                np.random.shuffle(tmpsels)
            if len(tmpids)%batch_size == 0:
                nums = len(tmpids)/batch_size
            else:
                nums = len(tmpids)/batch_size + 1
            for time in xrange(nums):
                start = time*batch_size
                end = (time+1)*batch_size
                sel = tmpsels[start:end].tolist()
                if not self.is_test:
                    yield torch.LongTensor(tmpdata[sel]),torch.LongTensor(tmpids[sel]),torch.LongTensor(tmplabel[sel])
                    #yield tmpdata[sel],tmplabel[sel],tmpids[sel]
                else:
                    yield torch.LongTensor(tmpdata[sel]), torch.LongTensor(tmpids[sel])
                    #yield tmpdata[sel], tmpids[sel]

    def _get_item2id_id2item(self,corpus):
        item2idx = {_PAD_:0}
        sen_list = corpus.values.tolist()
        for sen in sen_list:
            for word in sen.split():
                if word not in item2idx:
                    item2idx[word] = len(item2idx)
        idx2item = {v:k for k,v in item2idx.items()}
        return item2idx,idx2item

    @property
    def _temp_file(self):
        combine_str = "all_inter_files_item2idx_idx2item_item_embed_q2idvec"
        hash_object = hashlib.md5(combine_str.encode("utf-8"))
        hexcode = hash_object.hexdigest()
        return "./temp/%s_%s.pkl"%(combine_str,hexcode)

    def get_item_embed_tensor(self,space):
        return torch.Tensor(self.item_embed[space])



class MyDataLoader(object):
    def __init__(self,bucket_num,batch_size,train_test,data_space,shuffle=True,pad_prefix=True):
        self.batch_size = batch_size
        self.bucket_num = bucket_num
        self.train_test = train_test
        assert self.train_test in ["train","test"]
        self.data_space = data_space
        assert self.data_space in ["words","chars"]
        self.pad_prefix = pad_prefix
        self.shuffle = shuffle
        print("preprocessing...")
        self._preprocess()
        print("finish.")

    @property
    def _temp_file(self):
        combine_str = "%s_%s_%s_%s"%(self.train_test,self.data_space,self.bucket_num,self.pad_prefix)
        hash_object = hashlib.md5(combine_str.encode("utf-8"))
        hexcode = hash_object.hexdigest()
        return "./temp/%s_%s.pkl"%(combine_str,hexcode)

    def _preprocess(self):
        if os.path.exists(self._temp_file):
            print("detect cached intermediate files...loading...")
            all_cached = pickle.load(open(self._temp_file,"rb"))
            self.item2idx = all_cached["item2idx"]
            self.idx2item = all_cached["idx2item"]
            self.buckets = all_cached["buckets"]
            self.bounds = all_cached["bounds"]
            self.bucket_idx_vectors = all_cached["bucket_idx_vectors"]
        else:
            self._generate_inter_files()

    def _generate_inter_files(self):
        print("loading question_df...")
        question_df = DataSet.load_all_questions()
        corpus = question_df[self.data_space]
        print("generating item2idx...")
        sen_list = corpus.values.tolist()
        self.item2idx = {_PAD_:0}
        for sen in sen_list:
            for word in sen.split():
                if word not in self.item2idx:
                    self.item2idx[word] = len(self.item2idx)
        print("generating idx2item...")
        self.idx2item = {v:k for k,v in self.item2idx.items()}

        print("load %s data..."%(self.train_test))
        if self.train_test=="train":
            self.data_set = DataSet.load_train()
        else:
            self.data_set = DataSet.load_test()

        if self.data_space == "words":
            q1 = self.data_set["word_len_q1"]
            q2 = self.data_set["word_len_q2"]
        else:
            q1 = self.data_set["char_len_q1"]
            q2 = self.data_set["char_len_q2"]

        print("bucketing...")
        q_pair = list(zip(q1,q2))
        bucket = GreedyBucket()
        fit_res = bucket.fit(q_pair)
        self.buckets,self.bounds = bucket.get_split_results(fit_res,self.bucket_num)
        #print("len of self.bounds",len(self.bounds))
        print("generating id vectors...")
        data_set_id_vectors = []
        for ind in range(self.data_set.shape[0]):
            cur_row = self.data_set.iloc[ind]
            cur_q1 = cur_row["q1"]
            cur_q1_items = question_df.loc[cur_q1][self.data_space].split()
            cur_q1_inds = [self.item2idx[x] for x in cur_q1_items]

            cur_q2 = cur_row["q2"]
            cur_q2_items = question_df.loc[cur_q2][self.data_space].split()
            cur_q2_inds = [self.item2idx[x] for x in cur_q2_items]

            cur_bound = self.bounds[ind]
            q1_pad_len = cur_bound - len(cur_q1_inds)
            q2_pad_len = cur_bound - len(cur_q2_inds)

            if self.pad_prefix:
                cur_q1_padded = [0]*q1_pad_len+cur_q1_inds
                cur_q2_padded = [0]*q2_pad_len+cur_q2_inds
            else:
                cur_q1_padded = cur_q1_inds+[0]*q1_pad_len
                cur_q2_padded = cur_q2_inds+[0]*q2_pad_len
            cur_pair_padded = cur_q1_padded + cur_q2_padded
            data_set_id_vectors.append(cur_pair_padded)
        data_set_id_vectors = np.array(data_set_id_vectors)

        print("generating bucket_idx_vectors...")
        self.bucket_idx_vectors = {}
        for b,id_list in self.buckets.items():
            tmp = {}
            if self.train_test == "train":
                tmplabels = self.data_set["label"].iloc[id_list].values
                tmp["label"] = tmplabels
            tmpdata = np.array(data_set_id_vectors[id_list].tolist())
            tmp["data"] = tmpdata
            self.bucket_idx_vectors[b] = tmp

        print("finish generating inter files.")
        print("begin caching..")
        all_cached = {}
        all_cached["item2idx"] =  self.item2idx
        all_cached["idx2item"] =  self.idx2item
        all_cached["buckets"]  =  self.buckets
        all_cached["bounds"]   =  self.bounds
        all_cached["bucket_idx_vectors"] = self.bucket_idx_vectors
        try:
            os.makedirs("./temp")
        except:
            pass
        pickle.dump(all_cached,open(self._temp_file,"wb"))
        print("finish caching")

    def get_data_iterator(self):
        bucketkeys = list(self.buckets.keys())
        np.random.shuffle(bucketkeys)
        for b in bucketkeys:
            tmp = self.bucket_idx_vectors[b]
            if self.train_test == "train":
                tmpdataset = PaddedTensorDataset(tmp["data"],tmp["label"])
            else:
                tmpdataset = PaddedTensorDataset(tmp["data"],np.array(self.buckets[b]))
            for data_and_target_or_orig_ in DataLoader(tmpdataset,batch_size=self.batch_size,shuffle=self.shuffle):
                yield data_and_target_or_orig_

    def get_item_embeddings(self):
        term_embed = DataSet.load_term_embed(self.data_space)
        embed_size = term_embed.shape[1]
        pad_embed = np.array([0]*embed_size).reshape(1,-1)
        all_embeding = np.vstack([pad_embed,term_embed])
        all_index = [_PAD_]+term_embed.index.values.tolist()
        all_embeding_df = pd.DataFrame(data=all_embeding,index=all_index)
        sort_word = [self.idx2item[i] for i in range(len(self.idx2item))]
        return all_embeding_df.loc[sort_word].values







if __name__ == "__main__":
    # train = DataSet.load_train()
    # a = zip(train["word_len_q1"],train["word_len_q2"])[:1000]
    # bucket = GreedyBucket()
    # fitres = bucket.fit(a)
    # bucket,bounds = bucket.get_split_results(fitres,5)
    # print bucket,bounds
    dl = DataGenerator()
    train = DataSet.load_train()
    from sklearn.model_selection import train_test_split

    xtr, xte = train_test_split(train, test_size=0.33)
    dl.prepare(xtr[:10], 1, "words", True, False)
    tr_g = dl.get_data_generator(True, 20)
    for d,i,l in tr_g:
        print(d)
        print(i)
        print(l)
