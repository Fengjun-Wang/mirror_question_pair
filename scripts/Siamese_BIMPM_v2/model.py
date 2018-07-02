# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
sys.path.append("../")
import torch
import copy
import time
import os
import math
import pandas as pd
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from past.builtins import xrange
from feature_engineering.util import DataSet
from sklearn.model_selection import train_test_split
from itertools import ifilter
from sklearn.model_selection import KFold
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence,pack_sequence,pad_sequence
from torch import nn as nn, autograd
from torch.nn import functional as F
from torch.optim import lr_scheduler
from data import DataGenerator

class BIMPM(nn.Module):
    def __init__(self,
                 pre_trained_embedding,
                 is_freeze,
                 crl_hidden_size,
                 crl_number_layers,
                 crl_drop_p,
                 al_hidden_size,
                 al_number_layers,
                 al_drop_p,
                 number_perspective,
                 linear1_hidden_size,
                 linear2_hidden_size,
                 linear1_drop_p,
                 linear2_drop_p,
                 input_drop_p,
                 matching_layer_drop_p
                 ):
        super(BIMPM,self).__init__()
        self.crl_input_size = pre_trained_embedding.size(1)
        self.is_freeze = is_freeze
        self.crl_hidden_size = crl_hidden_size
        self.crl_number_layers = crl_number_layers
        self.crl_drop_p = crl_drop_p
        self.al_hidden_size = al_hidden_size
        self.al_number_layers = al_number_layers
        self.al_drop_p = al_drop_p
        self.number_perspective = number_perspective
        self.linear1_hidden_size = linear1_hidden_size
        self.linear2_hidden_size = linear2_hidden_size
        self.linear1_drop_p = linear1_drop_p
        self.linear2_drop_p = linear2_drop_p
        self.input_drop_p = input_drop_p
        self.matching_layer_drop_p = matching_layer_drop_p
        self.device  = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ##---Embedding layer---##
        self.nn_Embedding = nn.Embedding.from_pretrained(pre_trained_embedding, freeze=self.is_freeze)

        ##---Context representation layer---##
        self.crl = nn.LSTM(input_size = self.crl_input_size,
                           hidden_size = self.crl_hidden_size,
                           num_layers = self.crl_number_layers,
                           bidirectional = True,
                           dropout = self.crl_drop_p,
                           batch_first=True)

        ##---Aggregation layer---##
        self.al = nn.LSTM(input_size = 2*self.number_perspective,##TODO
                          hidden_size = self.al_hidden_size,
                          num_layers = self.al_number_layers,
                          bidirectional = True,
                          dropout = self.al_drop_p,
                          batch_first = True)

        ##---Matching Layers---##
        for i in xrange(1,9):
            setattr(self,"MW%s"%(i),nn.Parameter(torch.randn(self.number_perspective,self.crl_hidden_size)))


        ##---Input dropout---##
        self.input_drop = nn.Dropout(p=self.input_drop_p)
        self.matching_drop = nn.Dropout(p=self.matching_layer_drop_p)
        self.linear1_drop = nn.Dropout(p=self.linear1_drop_p)
        self.linear2_drop = nn.Dropout(p=self.linear2_drop_p)

        ##---Linear layer 1,2,3---##
        self.linear1 = nn.Linear(in_features = self.al_hidden_size*4,
                                 out_features = self.linear1_hidden_size)
        self.linear2 = nn.Linear(in_features = self.linear1_hidden_size,
                                 out_features = self.linear2_hidden_size)
        self.linear3 = nn.Linear(in_features=self.linear2_hidden_size,
                                 out_features=1)


    def init_weight_keras(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if (('bias_ih' in name) or ("bias_hh" in name)))
        #nn.init.uniform(self.embed.weight.data, a=-0.5, b=0.5)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)
            n = t.size(0)
            start, end = n // 4, n // 2
            t[start:end].fill_(1.)

    def sort(self,input_tensor):
        input_lengths = torch.LongTensor([torch.max(input_tensor[i, :].data.nonzero()) + 1 for i in xrange(input_tensor.size(0))])
        input_lengths, perm_idx = input_lengths.sort(0, descending=True)
        _,reverse_perm_idx = perm_idx.sort(0)
        input_seqs = input_tensor[perm_idx][:, :input_lengths.max()]
        return input_seqs, input_lengths, perm_idx, reverse_perm_idx


    def full_matching(self,q1_NLC,q2_NC,W):
        """
        :param q1_NLC:
        :param q2_NC:
        :return:NLP(p is number of perspective)
        """
        q1_NL1C = q1_NLC.unsqueeze(2)
        W_11PC = W.unsqueeze(0).unsqueeze(0)
        q1_NLPC = q1_NL1C*W_11PC
        q2_N11C = q2_NC.unsqueeze(1).unsqueeze(1)
        q2_N1PC = q2_N11C*W_11PC
        q1_mm_q2_NLPC = q1_NLPC*q2_N1PC
        q1_mm_q2_NLP = q1_mm_q2_NLPC.sum(dim=3)
        q1_NLPC_norm = q1_NLPC.norm(p=2,dim=3)
        q2_N1PC_norm = q2_N1PC.norm(p=2,dim=3)
        return q1_mm_q2_NLP/((q1_NLPC_norm*q2_N1PC_norm).clamp(min=1e-8))


    def maxpool_matching(self,q1_NLC,q2_NLC,W,q2_lengths):
        compare_L = torch.zeros(q2_NLC.size(1),q1_NLC.size(0),q1_NLC.size(1),self.number_perspective).to(self.device)
        for l in xrange(q2_NLC.size(1)):
            tmp_h_NC = q2_NLC[:,l,:]
            compare_L[l] = self.full_matching(q1_NLC,tmp_h_NC,W)
        res = torch.zeros(q1_NLC.size(0),q1_NLC.size(1),self.number_perspective).to(self.device)
        for i,l in enumerate(q2_lengths):
            res[i] = compare_L[:l,i].max(dim=0)[0]
        return res


    def attentive_matching(self,q1_NLC,q2_NLC,W):
        q1_NLC_norm = q1_NLC/(q1_NLC.norm(p=2,dim=2,keepdim=True).clamp(min=1e-8))
        q2_NLC_norm = q2_NLC/(q2_NLC.norm(p=2,dim=2,keepdim=True).clamp(min=1e-8))
        q2_NCL_norm = q2_NLC_norm.transpose(1,2)
        q1_q2_NLL = q1_NLC_norm.bmm(q2_NCL_norm)
        q1_q2_NLL_norm = q1_q2_NLL/q1_q2_NLL.sum(dim=2,keepdim=True).clamp(min=1e-8)
        q1_L_mean_NLC = q1_q2_NLL_norm.bmm(q2_NLC)

        q1_NLC_exp_NL1C = q1_NLC.unsqueeze(2)
        W_exp_11PC = W.unsqueeze(0).unsqueeze(0)
        q1_NLPC = q1_NLC_exp_NL1C * W_exp_11PC

        q1_L_mean_NLPC = q1_L_mean_NLC.unsqueeze(2)* W_exp_11PC ##NL1C * 11PC

        q1_L_mean_NLP_sum = (q1_NLPC*q1_L_mean_NLPC).sum(dim=3)##NLP
        q1_NLPC_norm = q1_NLPC.norm(p=2,dim=3)##NLP
        q1_L_mean_NLPC_norm = q1_L_mean_NLPC.norm(p=2,dim=3)##NLP

        return q1_L_mean_NLP_sum/((q1_NLPC_norm*q1_L_mean_NLPC_norm).clamp(min=1e-8))

    def max_attentive_matching(self,q1_NLC,q2_NLC,W,q2_lengths):
        q1_NLC_norm = q1_NLC/(q1_NLC.norm(p=2,dim=2,keepdim=True).clamp(min=1e-8))
        q2_NLC_norm = q2_NLC/(q2_NLC.norm(p=2,dim=2,keepdim=True).clamp(min=1e-8))
        q2_NCL_norm = q2_NLC_norm.transpose(1,2)
        q1_q2_NLL_cosine = q1_NLC_norm.bmm(q2_NCL_norm)
        res = torch.zeros(q1_NLC.size(0),q1_NLC.size(1),q1_NLC.size(2)).to(self.device)##NLC
        for i,l in enumerate(q2_lengths):
            tmp = q1_q2_NLL_cosine[i,:,:l]
            inds = tmp.max(dim=1)[1]
            res[i] = q2_NLC[i,inds,:]

        q1_NLC_exp_NL1C = q1_NLC.unsqueeze(2)
        W_exp_11PC = W.unsqueeze(0).unsqueeze(0)
        q1_NLPC = q1_NLC_exp_NL1C * W_exp_11PC

        q1_max_NLPC = res.unsqueeze(2)* W_exp_11PC ##NL1C * 11PC

        q1_max_NLP_sum = (q1_NLPC*q1_max_NLPC).sum(dim=3)##NLP
        q1_NLPC_norm = q1_NLPC.norm(p=2,dim=3)##NLP
        q1_max_NLPC_norm = q1_max_NLPC.norm(p=2,dim=3)##NLP

        return q1_max_NLP_sum/((q1_NLPC_norm*q1_max_NLPC_norm).clamp(min=1e-8))


    def forward(self,input):
        q1,q2  = torch.chunk(input, 2, dim=1) ## Split the question pairs
        q1, q1_lens, q1_perm_idx, q1_reverse_order_indx = self.sort(q1)
        q2, q2_lens, q2_perm_idx, q2_reverse_order_indx = self.sort(q2)
        q1_pad_embed = self.nn_Embedding(q1)##NLC
        q2_pad_embed = self.nn_Embedding(q2)##NLC
        q1_embed = self.input_drop(q1_pad_embed)
        q2_embed = self.input_drop(q2_pad_embed)
        q1_pack_pad_seq_embed = pack_padded_sequence(q1_embed, batch_first=True, lengths=q1_lens)
        q2_pack_pad_seq_embed = pack_padded_sequence(q2_embed, batch_first=True, lengths=q2_lens)
        ##Q1
        q1_out,q1_hidden = self.crl(q1_pack_pad_seq_embed)
        pad_q1_out = pad_packed_sequence(q1_out, batch_first=True)
        q1h,_ = q1_hidden
        pad_q1_forward,pad_q1_back = torch.chunk(pad_q1_out[0],2,dim=2)##NLC
        h_q1_forward = q1h[-2]
        h_q1_back = q1h[-1]
        pad_q1_forward_orig = pad_q1_forward[q1_reverse_order_indx]
        pad_q1_back_orig = pad_q1_back[q1_reverse_order_indx]
        h_q1_forward_orig = h_q1_forward[q1_reverse_order_indx]
        h_q1_back_orig = h_q1_back[q1_reverse_order_indx]
        q1_lens_orig = q1_lens[q1_reverse_order_indx]
        ##Q2
        q2_out,q2_hidden = self.crl(q2_pack_pad_seq_embed)
        pad_q2_out = pad_packed_sequence(q2_out, batch_first=True)
        q2h,_ = q2_hidden
        pad_q2_forward,pad_q2_back = torch.chunk(pad_q2_out[0],2,dim=2)##NLC
        h_q2_forward = q2h[-2]
        h_q2_back = q2h[-1]
        pad_q2_forward_orig = pad_q2_forward[q2_reverse_order_indx]
        pad_q2_back_orig = pad_q2_back[q2_reverse_order_indx]
        h_q2_forward_orig = h_q2_forward[q2_reverse_order_indx]
        h_q2_back_orig = h_q2_back[q2_reverse_order_indx]
        q2_lens_orig = q2_lens[q2_reverse_order_indx]

        #q1_for_full_matching = self.full_matching(pad_q1_forward_orig,h_q2_forward_orig,self.MW1)
        #q1_back_full_matching = self.full_matching(pad_q1_back_orig,h_q2_back_orig,self.MW2)
        q1_for_maxpool_matching = self.maxpool_matching(pad_q1_forward_orig,pad_q2_forward_orig,self.MW3,q2_lens_orig)
        q1_back_maxpool_matching = self.maxpool_matching(pad_q1_back_orig,pad_q2_back_orig,self.MW4,q2_lens_orig)
        #q1_for_att_matching = self.attentive_matching(pad_q1_forward_orig,pad_q2_forward_orig,self.MW5)
        #q1_back_att_matching = self.attentive_matching(pad_q1_back_orig,pad_q2_back_orig,self.MW6)
        #q1_for_maxatt_matching = self.max_attentive_matching(pad_q1_forward_orig,pad_q2_forward_orig,self.MW7,q2_lens_orig)
        #q1_back_maxatt_matching = self.max_attentive_matching(pad_q1_back_orig,pad_q2_back_orig,self.MW8,q2_lens_orig)

        #q2_for_full_matching = self.full_matching(pad_q2_forward_orig,h_q1_forward_orig,self.MW1)
        #q2_back_full_matching = self.full_matching(pad_q2_back_orig,h_q1_back_orig,self.MW2)
        q2_for_maxpool_matching = self.maxpool_matching(pad_q2_forward_orig,pad_q1_forward_orig,self.MW3,q1_lens_orig)
        q2_back_maxpool_matching = self.maxpool_matching(pad_q2_back_orig,pad_q1_back_orig,self.MW4,q1_lens_orig)
        #q2_for_att_matching = self.attentive_matching(pad_q2_forward_orig,pad_q1_forward_orig,self.MW5)
        #q2_back_att_matching = self.attentive_matching(pad_q2_back_orig,pad_q1_back_orig,self.MW6)
        #q2_for_maxatt_matching = self.max_attentive_matching(pad_q2_forward_orig,pad_q1_forward_orig,self.MW7,q1_lens_orig)
        #q2_back_maxatt_matching = self.max_attentive_matching(pad_q2_back_orig,pad_q1_back_orig,self.MW8,q1_lens_orig)

        q1_agg = torch.cat([#q1_for_full_matching,
                            #q1_back_full_matching,
                            q1_for_maxpool_matching,
                            q1_back_maxpool_matching,
                            #q1_for_att_matching,
                            #q1_back_att_matching,
                            #q1_for_maxatt_matching,
                            #q1_back_maxatt_matching
                             ],dim=2) ##NXLX8P
        #print("q1_agg")
        #print(q1_agg.size())
        q2_agg = torch.cat([
            #q2_for_full_matching,
            #q2_back_full_matching,
            q2_for_maxpool_matching,
            q2_back_maxpool_matching,
            #q2_for_att_matching,
            #q2_back_att_matching,
            #q2_for_maxatt_matching,
            #q2_back_maxatt_matching
        ],dim=2)##NXLX8P
        #print("q2_agg")
        #print(q2_agg.size())
        q1_agg_order = q1_agg[q1_perm_idx]
        q2_agg_order = q2_agg[q2_perm_idx]
        q1_agg_order = self.matching_drop(q1_agg_order)
        q2_agg_order = self.matching_drop(q2_agg_order)
        q1_pack_pad_agg_order = pack_padded_sequence(q1_agg_order, batch_first=True, lengths=q1_lens)
        q2_pack_pad_agg_order = pack_padded_sequence(q2_agg_order, batch_first=True, lengths=q2_lens)

        q1_agout,q1_aghidden = self.al(q1_pack_pad_agg_order)
        q1agh,_ = q1_aghidden

        q2_agout,q2_aghidden = self.al(q2_pack_pad_agg_order)
        q2agh,_ = q2_aghidden


        q1_agencode = torch.cat((q1agh[-2], q1agh[-1]), dim=1)
        q2_agencode = torch.cat((q2agh[-2], q2agh[-1]), dim=1)
        q1_encode_reverse = q1_agencode[q1_reverse_order_indx]
        q2_encode_reverse = q2_agencode[q2_reverse_order_indx]
        q_pair_encode_q12= torch.cat((q1_encode_reverse,q2_encode_reverse),dim=1)
        hid1 = self.linear1_drop(F.relu(self.linear1(q_pair_encode_q12)))
        hid2 = self.linear2_drop(F.relu(self.linear2(hid1)))
        out = self.linear3(hid2)
        return out


class Model(object):
    def __init__(self,name,model):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.name = name
        try:
            os.makedirs(self.name)
        except:
            pass

    def train(self,train_dg,valid_dg,criterion, optimizer, scheduler, num_epochs, early_stop_rounds):
        if early_stop_rounds==-1: ##If it is -1, then cancel early stopping
            early_stop_rounds = sys.maxsize
        since = time.time()
        criterion = criterion.to(self.device)
        dataset_sizes = {"train":len(train_dg),"val":len(valid_dg)}
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        best_loss = sys.maxsize
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        early_stop_epochs = 0
        early_stop_flag = False
        for epoch in xrange(num_epochs):
            if early_stop_flag:
                break
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode
                running_loss = 0.0
                running_corrects = 0
                # Iterate over data.
                sample_num = 0
                data_loader = train_dg.get_data_generator() if phase=="train" else valid_dg.get_data_generator()
                for inputs, ids, labels in data_loader: ##inputs and labels are dim 2 and ids are dim 1
                    sample_num += len(inputs)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        preds = outputs>0
                        preds = preds.type(torch.cuda.FloatTensor)
                        loss = criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            #torch.nn.utils.clip_grad_norm(self.model.parameters(), 10)
                            optimizer.step()
                    running_loss += loss.item() * inputs.size(0) ## criteria size_average is True default
                    running_corrects += torch.sum(preds.data == labels.data) ## running_corrects are tensor dim 0
                epoch_loss = running_loss / dataset_sizes[phase] ##average loss
                epoch_acc = running_corrects.double().item() / dataset_sizes[phase]
                print('{} Loss: {:.4f} Acc: {:.4f} Num: {}'.format(
                    phase, epoch_loss, epoch_acc,sample_num))
                ##Record the train and validation loss and accuracy for analysis
                if phase=="train":
                    self.train_loss.append(epoch_loss)
                    self.train_acc.append(epoch_acc)
                else:
                    self.val_loss.append(epoch_loss)
                    self.val_acc.append(epoch_acc)
                    if epoch_loss < best_loss:
                        early_stop_epochs = 0
                        best_loss = epoch_loss
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                    else:
                        early_stop_epochs+=1
                        if early_stop_epochs>=early_stop_rounds:
                            early_stop_flag = True
            print()
        time_elapsed = time.time() - since
        if early_stop_flag:
            print('Training complete due to early stopping in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
        else:
            print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_loss))
        print('Corresponding val Acc: {:4f}'.format(best_acc))
        # load best model weights
        self.model.load_state_dict(best_model_wts)
        torch.save(self.model.state_dict(),self.name+"/siamese_CNN_best_pars.pth")
        return self.model

    def predict(self,data_dg):
        self.model.eval()
        preds = torch.zeros(len(data_dg),1).to(self.device)
        with torch.no_grad():
            for inputs, ids, _ in data_dg.get_data_generator():
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = F.sigmoid(outputs)
                preds[ids,:] = probs
        return preds.squeeze(dim=1).cpu()

    def save_plot_(self): ##TODO add save function
        plt.figure(figsize=(20, 8))
        l1, = plt.plot(self.train_loss, "r-")
        l2, = plt.plot(self.val_loss, "g-")
        plt.legend([l1, l2], ["train_loss", "valid_loss"])
        plt.grid()
        plt.savefig(self.name+"/%s_loss_trend.jpg"%(self.name))
        plt.figure(figsize=(20, 8))
        l1, = plt.plot(self.train_acc, "r-")
        l2, = plt.plot(self.val_acc, "g-")
        plt.legend([l1, l2], ["train_acc", "valid_acc"])
        plt.grid()
        plt.savefig(self.name+"/%s_acc_trend.jpg"%(self.name))

        res = pd.DataFrame({"train_loss":self.train_loss,
                           "val_loss":self.val_loss,
                           "train_acc":self.train_acc,
                           "val_acc":self.val_acc})
        res[["train_acc","val_acc","train_loss","val_loss"]].to_csv(self.name+"/%s_train_process.csv"%(self.name),index=False)


##TODO Put parameters here
##--------------parameters-------------------##
is_freeze = False
crl_hidden_size = 100
crl_number_layers = 1
crl_drop_p = 0.0
al_hidden_size = 100
al_number_layers = 1
al_drop_p = 0.0
number_perspective = 10
linear1_hidden_size = 200
linear2_hidden_size = 100
linear1_drop_p = 0.2
linear2_drop_p = 0.2
input_drop_p = 0.4
matching_layer_drop_p = 0.3
space = "words"
batch_size = 64 ##
folder = 5
early_stop = 10
LR = 0.001
Gamma = 0.99
num_epochs = 150
version = "v01"
##--------------parameters-------------------##

def train_main():

    train_name = version+"_sm"
    ##--------------parameters-------------------##

    train_df = DataSet.load_train()
    xtr_df, xval_df = train_test_split(train_df, test_size=0.20)
    test_df = DataSet.load_test()
    ### Generate data generator
    train_dg = DataGenerator(data_df=xtr_df,space=space,bucket_num=5,batch_size=batch_size,is_prefix_pad=False,is_shuffle=True,is_test=False)
    val_dg = DataGenerator(data_df=xval_df,space=space,bucket_num=5,batch_size=512,is_prefix_pad=False,is_shuffle=False,is_test=False)
    test_dg = DataGenerator(data_df=test_df,space=space,bucket_num=5,batch_size=512,is_prefix_pad=False,is_shuffle=False,is_test=True)
    ### Must do prepare before using
    train_dg.prepare()
    val_dg.prepare()
    test_dg.prepare()
    ### load word embedding, can use train_df, val_dg or test_dg
    item_embed = train_dg.get_item_embed_tensor(space)
    ### Initialize network
    bimpm = BIMPM(
                 pre_trained_embedding = item_embed,
                 is_freeze = is_freeze,
                 crl_hidden_size = crl_hidden_size,
                 crl_number_layers = crl_number_layers,
                 crl_drop_p = crl_drop_p,
                 al_hidden_size = al_hidden_size,
                 al_number_layers = al_number_layers,
                 al_drop_p = al_drop_p,
                 number_perspective = number_perspective,
                 linear1_hidden_size = linear1_hidden_size,
                 linear2_hidden_size = linear2_hidden_size,
                 linear1_drop_p = linear1_drop_p,
                 linear2_drop_p = linear2_drop_p,
                 input_drop_p = input_drop_p,
                 matching_layer_drop_p = matching_layer_drop_p)
    bimpm.init_weight_keras() ##TODO Whether initialize customised weights as Keras
    ### Initialize model using network
    bimpm_model = Model(train_name,bimpm)
    criteria = nn.BCEWithLogitsLoss()
    optimizer_ft = optim.Adam(ifilter(lambda p: p.requires_grad, bimpm.parameters()), lr=LR)
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer_ft, gamma=Gamma)
    ### Train
    bimpm_model.train(train_dg=train_dg,
                        valid_dg=val_dg,
                        criterion=criteria,
                        optimizer=optimizer_ft,
                        scheduler=exp_lr_scheduler,
                        num_epochs=num_epochs,
                        early_stop_rounds=early_stop)
    bimpm_model.save_plot_()
    preds = bimpm_model.predict(test_dg).numpy()
    preds = pd.DataFrame({"y_pre":preds})
    preds.to_csv(version+"_submission_sm.csv",index=False)

def cv_main():
    kf = KFold(n_splits=folder,shuffle=True,random_state=19920618)
    all_train_df = DataSet.load_train()
    test_df = DataSet.load_test()
    test_dg = DataGenerator(data_df=test_df,space=space,bucket_num=5,batch_size=256,is_prefix_pad=False,is_shuffle=False,is_test=True)
    print("prepare test data generator")
    test_dg.prepare()
    item_embed = test_dg.get_item_embed_tensor(space)
    train_eval = np.zeros(len(all_train_df))
    test_eval = np.zeros((len(test_df),folder))
    for i,(train_index,val_index) in enumerate(kf.split(all_train_df)):
        print()
        train_name = version + "_cv_%s"%(i)
        xtr_df = all_train_df.iloc[train_index]
        xval_df = all_train_df.iloc[val_index]
        train_dg = DataGenerator(data_df=xtr_df, space=space, bucket_num=5, batch_size=batch_size, is_prefix_pad=False,
                                 is_shuffle=True, is_test=False)
        val_dg = DataGenerator(data_df=xval_df, space=space, bucket_num=5, batch_size=256, is_prefix_pad=False,
                               is_shuffle=False, is_test=False)
        print("prepare train data generator, cv_%s"%i)
        train_dg.prepare()
        print("prepare val data generator, cv_%s" % i)
        val_dg.prepare()
        bimpm = BIMPM(
            pre_trained_embedding=item_embed,
            is_freeze=is_freeze,
            crl_hidden_size=crl_hidden_size,
            crl_number_layers=crl_number_layers,
            crl_drop_p=crl_drop_p,
            al_hidden_size=al_hidden_size,
            al_number_layers=al_number_layers,
            al_drop_p=al_drop_p,
            number_perspective=number_perspective,
            linear1_hidden_size=linear1_hidden_size,
            linear2_hidden_size=linear2_hidden_size,
            linear1_drop_p=linear1_drop_p,
            linear2_drop_p=linear2_drop_p,
            input_drop_p=input_drop_p,
            matching_layer_drop_p=matching_layer_drop_p)

        bimpm.init_weight_keras()  ##TODO Whether to initialize customised weights as Keras
        bimpm_model = Model(train_name, bimpm)
        criteria = nn.BCEWithLogitsLoss()
        optimizer_ft = optim.Adam(ifilter(lambda p: p.requires_grad, bimpm.parameters()), lr=LR)
        exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer_ft, gamma=Gamma)
        ### Train
        bimpm_model.train(train_dg=train_dg,
                            valid_dg=val_dg,
                            criterion=criteria,
                            optimizer=optimizer_ft,
                            scheduler=exp_lr_scheduler,
                            num_epochs=num_epochs,
                            early_stop_rounds=early_stop)
        bimpm_model.save_plot_()
        val_pred = bimpm_model.predict(val_dg).numpy()
        train_eval[val_index] = val_pred
        test_preds = bimpm_model.predict(test_dg).numpy()
        test_eval[:,i] = test_preds
    train_pred_df = pd.DataFrame({version+"_train_pred_cv":train_eval})
    train_pred_df.to_csv(version+"_train_pred_cv.csv",index=False)
    test_pred_df = pd.DataFrame(test_eval,columns=[version+"_test_pred_cv_%s"%(i) for i in xrange(folder)])
    test_pred_df["y_pre"] = test_pred_df.mean(axis=1)
    test_pred_df.to_csv(version+"_test_pred_cv.csv",index=False)
    test_pred_df[["y_pre"]].to_csv(version+"_submission_cv.csv",index=False)

##BCEWithLogitsLoss
#https://discuss.pytorch.org/t/how-to-initiate-parameters-of-layers/1460
if __name__ == "__main__":
    #train_main()
    cv_main()
