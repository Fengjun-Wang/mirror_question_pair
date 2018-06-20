# -*- coding: utf-8 -*-
import sys
sys.path.append("../")
from feature_engineering.util import DataSet

class GreedyBucket(object):
    def fit(self,list_pair_len):
        list_pair_len_max = map(max, list_pair_len)
        max_and_pair_list = zip(list_pair_len_max,list_pair_len)
        max_cost_list = map(lambda x:(x[0],2*x[0]-x[1][0]-x[1][1]),max_and_pair_list)
        self.sort_max_cost_list = sorted(enumerate(max_cost_list),key=lambda x:x[1][0])
        return self.sort_max_cost_list

    def _binary_search_split_point(self,sort_max_cost_list,split_point):
        i = 0
        j = len(sort_max_cost_list)-1
        while i<=j:
            if sort_max_cost_list[(i+j)/2][1][0]>split_point:
                j -= 1
            else:
                i += 1
        return i-1

    def _get_possible_points(self,sort_max_cost_list):
        pss = set(map(lambda x:x[1][0],sort_max_cost_list))
        pss.remove(sort_max_cost_list[-1][1][0]) ## remove the last point, because spliting at last point means no spliting
        return pss

    def _find_split_points(self,sort_max_cost_list,num_bucket):
        num_splits = num_bucket - 1
        if num_splits == 0:
            return []
        else:
            possible_split_point = self._get_possible_points(sort_max_cost_list)
            mincost = sys.maxint
            for p_s in possible_split_point:
                split_index = self._binary_search_split_point(sort_max_cost_list,p_s)
                cost1 = 0
                cost2 = 0
                for i,item in enumerate(sort_max_cost_list):
                    if i<=split_index:
                        cost1 += 2*(p_s-item[1][0])+item[1][1]
                    else:
                        cost2 += 2*(sort_max_cost_list[-1][1][0]-item[1][0])+item[1][1]
                cur_cost = cost1+cost2
                if cur_cost < mincost:
                    mincost = cur_cost
                    record = p_s
                    min_c1 = cost1
                    min_c2 = cost2
                    min_index = split_index
            num_splits -= 1
            if num_splits%2==0:
                cur_res =  [(record,min_index)]
                next_len_1 = min_index+1
                next_len_2 = len(sort_max_cost_list)-next_len_1
                self._find_split_points(sort_max_cost_list[:min_index+1],num_splits/2+1)+\
                self._find_split_points(sort_max_cost_list[min_index+1:],num_splits/2+1)
            else:
                if min_c1<min_c2:
                    return [(record,min_index)]+\
                    self._find_split_points(sort_max_cost_list[:min_index],num_splits/2+1)+\
                    self._find_split_points(sort_max_cost_list[min_index+1:],num_splits/2+2)
                else:
                    return [(record,min_index)]+\
                    self._find_split_points(sort_max_cost_list[:min_index],num_splits/2+2)+\
                    self._find_split_points(sort_max_cost_list[min_index+1:],num_splits/2+1)

    def get_split_results(self,sort_max_cost_list,num_bucket):
        if num_bucket<1 or num_bucket>len(sort_max_cost_list):
            raise ValueError("'bucket_num' must be between 1 and len(sort_max_cost_listsort)")
        return [(sort_max_cost_list[-1][1][0],len(sort_max_cost_list)-1)]+self._find_split_points(sort_max_cost_list,num_bucket)
            



