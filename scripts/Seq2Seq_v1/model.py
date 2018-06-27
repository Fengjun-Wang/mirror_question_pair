# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
sys.path.append("../")
import torch
import copy
import time
import os
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

class Siamese_LSTM(nn.Module):
    def __init__(self,
                 pre_trained_embedding,
                 is_freeze,
                 hidden_size,
                 number_layers,
                 lstm_dropout_p,
                 bidirectional,
                 linear_hid_size,
                 linear_hid_drop_p,
                 input_drop_p):
        super(Siamese_LSTM,self).__init__()
        self.input_channel_len = pre_trained_embedding.size(1)
        self.is_freeze = is_freeze
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.lstm_dropout_p = lstm_dropout_p
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.linear_hid_size = linear_hid_size
        self.linear_hid_drop_p = linear_hid_drop_p
        self.input_drop_p = input_drop_p
        ## Init layers
        self.nn_Embedding = nn.Embedding.from_pretrained(pre_trained_embedding,freeze=self.is_freeze)
        self.lstm = nn.LSTM(input_size=self.input_channel_len,
                            hidden_size=self.hidden_size,
                            num_layers=self.number_layers,
                            batch_first=True,
                            dropout=self.lstm_dropout_p,
                            bidirectional=self.bidirectional)
        self.linear1 = nn.Linear(2*self.num_directions*hidden_size,self.linear_hid_size) ##out dim of q1 is 2*hidden(concat two hiddens), same for q2
        self.linear1_dropout = nn.Dropout(p=self.linear_hid_drop_p)
        self.linear2 = nn.Linear(self.linear_hid_size,1)
        self.input_dropout = nn.Dropout(p=self.input_drop_p)

    def init_hidden(self,batch_size):
        self.hidden = ( torch.zeros(self.number_layers*self.num_directions, batch_size, self.hidden_size),
                        torch.zeros(self.number_layers*self.num_directions, batch_size, self.hidden_size))

    def init_weights(self): ##TODO add weight initialization
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

    def sort(self,input_tensor):
        input_lengths = torch.LongTensor([torch.max(input_tensor[i, :].data.nonzero()) + 1 for i in xrange(input_tensor.size(0))])
        input_lengths, perm_idx = input_lengths.sort(0, descending=True)
        _,reverse_perm_idx = perm_idx.sort(0)
        input_seqs = input_tensor[perm_idx][:, :input_lengths.max()]
        return input_seqs,input_lengths,reverse_perm_idx

    def forward(self, input):
        q1,q2  = torch.chunk(input, 2, dim=1) ## Split the question pairs
        q1,q1_lens,q1_reverse_order_indx = self.sort(q1)
        q2,q2_lens,q2_reverse_order_indx = self.sort(q2)
        q1_pad_embed = self.nn_Embedding(q1) ##NxLxC
        q2_pad_embed = self.nn_Embedding(q2) ##NxLxC
        q1_embed = self.input_dropout(q1_pad_embed)
        q2_embed = self.input_dropout(q2_pad_embed)
        q1_pack_pad_seq_embed = pack_padded_sequence(q1_embed, batch_first=True, lengths=q1_lens)
        q2_pack_pad_seq_embed = pack_padded_sequence(q2_embed, batch_first=True, lengths=q2_lens)

        q1_out,q1_hidden = self.lstm(q1_pack_pad_seq_embed)
        q1h,q1c = q1_hidden

        q2_out,q2_hidden = self.lstm(q2_pack_pad_seq_embed)
        q2h,q2c = q2_hidden

        if self.bidirectional:
            q1_encode = torch.cat((q1h[-2],q1h[-1]),dim=1)
            q2_encode = torch.cat((q2h[-2],q2h[-1]),dim=1)
        else:
            q1_encode = q1h[-1]
            q2_encode = q2h[-1]
        q1_encode_reverse = q1_encode[q1_reverse_order_indx]
        q2_encode_reverse = q2_encode[q2_reverse_order_indx]

        q_pair_encode_q12= torch.cat((q1_encode_reverse,q2_encode_reverse),dim=1)
        q_pair_encode_q21 = torch.cat((q2_encode_reverse,q1_encode_reverse),dim=1)
        q_pair_encode = torch.cat((q_pair_encode_q12,q_pair_encode_q21),dim=0)
        h1 = self.linear1_dropout(F.relu(self.linear1(q_pair_encode)))
        out = self.linear2(h1)
        out1,out2 = torch.chunk(out,2,dim=0)
        return (out1+out2)/2


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
                        #preds = F.sigmoid(outputs)>0.5
                        preds = outputs>0
                        preds = preds.type(torch.cuda.FloatTensor)
                        loss = criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
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
        res.to_csv(self.name+"/%s_train_process.csv"%(self.name),index=False)


##TODO Put parameters here
##--------------parameters-------------------##
space = "words"
is_freeze = False ##TODO
hidden_size = 100 ##TODO 100->300->100
num_layers = 2
bidirectional = True
lstm_drop_p = 0.6
lstm_input_drop_p = 0.6
linear_hidden_size = 200##TODO 200->600->200
linear_hid_drop_p = 0.3
batch_size = 512
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
    siamese_lstm = Siamese_LSTM(pre_trained_embedding=item_embed,
                               is_freeze=is_freeze,
                               hidden_size=hidden_size,
                               number_layers=num_layers,
                               lstm_dropout_p=lstm_drop_p,
                               bidirectional=bidirectional,
                               linear_hid_size=linear_hidden_size,
                               linear_hid_drop_p=linear_hid_drop_p,
                               input_drop_p = lstm_input_drop_p)
    siamese_lstm.init_weights() ##TODO Whether initialize customised weights as Keras
    ### Initialize model using network
    siamese_model = Model(train_name,siamese_lstm)
    criteria = nn.BCEWithLogitsLoss()
    optimizer_ft = optim.Adam(ifilter(lambda p: p.requires_grad, siamese_lstm.parameters()), lr=LR)
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer_ft, gamma=Gamma)
    ### Train
    siamese_model.train(train_dg=train_dg,
                        valid_dg=val_dg,
                        criterion=criteria,
                        optimizer=optimizer_ft,
                        scheduler=exp_lr_scheduler,
                        num_epochs=num_epochs,
                        early_stop_rounds=early_stop)
    siamese_model.save_plot_()
    preds = siamese_model.predict(test_dg).numpy()
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
        siamese_lstm = Siamese_LSTM(pre_trained_embedding=item_embed,
                                    is_freeze=is_freeze,
                                    hidden_size=hidden_size,
                                    number_layers=num_layers,
                                    lstm_dropout_p=lstm_drop_p,
                                    bidirectional=bidirectional,
                                    linear_hid_size=linear_hidden_size,
                                    linear_hid_drop_p=linear_hid_drop_p,
                                    input_drop_p=lstm_input_drop_p)
        siamese_lstm.init_weights()  ##TODO Whether to initialize customised weights as Keras
        siamese_model = Model(train_name, siamese_lstm)
        criteria = nn.BCEWithLogitsLoss()
        optimizer_ft = optim.Adam(ifilter(lambda p: p.requires_grad, siamese_lstm.parameters()), lr=LR)
        exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer_ft, gamma=Gamma)
        ### Train
        siamese_model.train(train_dg=train_dg,
                            valid_dg=val_dg,
                            criterion=criteria,
                            optimizer=optimizer_ft,
                            scheduler=exp_lr_scheduler,
                            num_epochs=num_epochs,
                            early_stop_rounds=early_stop)
        siamese_model.save_plot_()
        val_pred = siamese_model.predict(val_dg).numpy()
        train_eval[val_index] = val_pred
        test_preds = siamese_model.predict(test_dg).numpy()
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
    train_main()
    #cv_main()
