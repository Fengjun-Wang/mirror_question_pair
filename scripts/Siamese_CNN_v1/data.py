# -*- coding: utf-8 -*-
import sys
import os
import pickle
import hashlib
sys.path.append("../")
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from feature_engineering.util import DataSet
_PAD_ = "_PAD_"
class GreedyBucket(object):
    def fit(self,list_pair_len):
        self.list_pair_len_max = map(max, list_pair_len)
        max_and_pair_list = zip(self.list_pair_len_max,list_pair_len)
        max_cost_list = map(lambda x:(x[0],2*x[0]-x[1][0]-x[1][1]),max_and_pair_list)
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
            bigger = sys.maxint
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
    def __init__(self, data_tensor, target_tensor):
        assert len(data_tensor)==len(target_tensor)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return len(self.data_tensor)

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
        print "preprocessing..."
        self._preprocess()
        print "finish."

    @property
    def _temp_file(self):
        combine_str = "%s_%s_%s_%s"%(self.train_test,self.data_space,self.bucket_num,self.pad_prefix)
        hash_object = hashlib.md5(combine_str.encode("utf-8"))
        hexcode = hash_object.hexdigest()
        return "./temp/%s_%s.pkl"%(combine_str,hexcode)

    def _preprocess(self):
        if os.path.exists(self._temp_file):
            print "detect cached intermediate files...loading..."
            all_cached = pickle.load(open(self._temp_file,"rb"))
            self.item2idx = all_cached["item2idx"]
            self.idx2item = all_cached["idx2item"]
            self.buckets = all_cached["buckets"]
            self.bounds = all_cached["bounds"]
            self.bucket_idx_vectors = all_cached["bucket_idx_vectors"]
            print "finish"
        else:
            self._generate_inter_files()

    def _generate_inter_files(self):
        print "loading question_df..."
        question_df = DataSet.load_all_questions()
        corpus = question_df[self.data_space]
        print "generating item2idx..."
        sen_list = map(lambda x:x.split(),corpus.values.tolist())
        self.item2idx = {_PAD_:0}
        for sen in sen_list:
            for word in sen:
                if word not in self.item2idx:
                    self.item2idx[word] = len(self.item2idx)
        print "generating idx2item..."
        self.idx2item = {v:k for k,v in self.item2idx.items()}

        print "load %s data..."%(self.train_test)
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

        print "bucketing..."
        q_pair = zip(q1,q2)
        bucket = GreedyBucket()
        fit_res = bucket.fit(q_pair)
        self.buckets,self.bounds = bucket.get_split_results(fit_res,self.bucket_num)

        print "generating id vectors..."
        data_set_id_vectors = []
        for ind in xrange(self.data_set.shape[0]):
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

        print "generating bucket_idx_vectors..."
        self.bucket_idx_vectors = {}
        for b,id_list in self.buckets.items():
            tmp = {}
            if self.train_test == "train":
                tmplabels = self.data_set["label"].iloc[id_list].values
                tmp["label"] = tmplabels
            tmpdata = np.array(data_set_id_vectors[id_list].tolist())
            tmp["data"] = tmpdata
            self.bucket_idx_vectors[b] = tmp

        print "finish generating inter files."
        print "begin caching.."
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
        print "finish caching"

    def get_data_iterator(self):
        bucketkeys = self.buckets.keys()
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
    dl = MyDataLoader(bucket_num=5,batch_size=2,train_test="train",data_space="words")
    dit = dl.get_data_iterator()
    for d,y in dit:
        print d
        print y
