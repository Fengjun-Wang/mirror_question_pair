# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
sys.path.append("../")
import torch
import copy
import time
import os
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from feature_engineering.util import DataSet
from sklearn.model_selection import train_test_split
from itertools import ifilter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch import nn as nn, autograd
from torch.nn import functional as F
from torch.optim import lr_scheduler
from data import DataGenerator
from sklearn.model_selection import KFold


class Similarity(nn.Module):  ##TODO Add similarity layer
    def __init__(self,dim):
        super(Similarity,self).__init__()
        self.matrix = nn.Parameter(torch.randn(dim,dim))
        #self.dropout = nn.Dropout(p=0.2)
    def forward(self, q1,q2):
        similarity = torch.mm(torch.mm(q1,self.matrix),q2.transpose(0,1))
        return similarity.diag().view(-1,1)

class Siamese_CNN(nn.Module):
    def __init__(self,pre_trained_embedding,is_freeze,output_channels,conv_p,h1_p): ##TODO Parameterize output_channels
        super(Siamese_CNN,self).__init__()
        self.nn_Embedding = nn.Embedding.from_pretrained(pre_trained_embedding,freeze=is_freeze)
        input_channel_len = pre_trained_embedding.size(1)
        #self.similarity = Similarity(3*output_channels) ##TODO Add similarity here
        self.convdrop = nn.Dropout(p=conv_p) ###
        self.h1_p = nn.Dropout(p=h1_p)
        #self.sim_p = nn.Dropout(p=sim_p)
        ##self.h2_p = nn.Dropout(p=h2_p)
        self.conv1d_size2 = nn.Conv1d(in_channels=input_channel_len,
                                      out_channels=output_channels,
                                      kernel_size=2,
                                      stride=1,
                                      padding=1)
        self.conv1d_size3 = nn.Conv1d(in_channels=input_channel_len,
                                      out_channels=output_channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        self.conv1d_size4 = nn.Conv1d(in_channels=input_channel_len,
                                      out_channels=output_channels,
                                      kernel_size=4,
                                      stride=1,
                                      padding=1)
        self.out_hidden1 = nn.Linear(12*output_channels,480) ##TODO Add 1 for similarity
        ##self.out_hidden2 = nn.Linear(300,150)
        self.out_put = nn.Linear(480,1)

    def forward(self, input):
        q1,q2  = torch.chunk(input, 2, dim=1) ## Split the question pairs
        q1_embed = self.nn_Embedding(q1).transpose(1,2) ##NxLxC -> NxCxL
        q2_embed = self.nn_Embedding(q2).transpose(1,2)

        q1_conv1 = F.relu(self.conv1d_size2(q1_embed))##NxCxL
        q1_pool1 = self.convdrop(q1_conv1.max(dim=2)[0]) ##NxC
        q1_conv2 = F.relu(self.conv1d_size3(q1_embed)) ##NxCxL
        q1_pool2 = self.convdrop(q1_conv2.max(dim=2)[0]) ##NxC
        q1_conv3 = F.relu(self.conv1d_size4(q1_embed)) ##NxCxL
        q1_pool3 = self.convdrop(q1_conv3.max(dim=2)[0]) ##NxC(100)
        q1_concat = torch.cat((q1_pool1,q1_pool2,q1_pool3),dim=1) ## Nx(c1+c2...)[300]

        q2_conv1 = F.relu(self.conv1d_size2(q2_embed))##NxCxL
        q2_pool1 = self.convdrop(q2_conv1.max(dim=2)[0]) ##NxC
        q2_conv2 = F.relu(self.conv1d_size3(q2_embed)) ##NxCxL
        q2_pool2 = self.convdrop(q2_conv2.max(dim=2)[0]) ##NxC
        q2_conv3 = F.relu(self.conv1d_size4(q2_embed)) ##NxCxL
        q2_pool3 = self.convdrop(q2_conv3.max(dim=2)[0]) ##NxC(100)
        q2_concat = torch.cat((q2_pool1,q2_pool2,q2_pool3),dim=1) ## Nx(c1+c2...)[300]
        #similarity = self.sim_p(self.similarity(q1_concat,q2_concat))
        q_concat = torch.cat((q1_concat,q2_concat,torch.abs(q1_concat-q2_concat),q1_concat*q2_concat),dim=1) ##Nx601
        h1 = self.h1_p(F.relu(self.out_hidden1(q_concat)))
        ##h2 = self.h2_p(F.relu(self.out_hidden2(h1)))
        outscore = self.out_put(h1)
        return outscore


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
        res[["train_acc","val_acc","train_loss","val_loss"]].to_csv(self.name+"/%s_train_process.csv"%(self.name),index=False)


##TODO Put parameters here
##--------------parameters-------------------##
space = "words"
is_freeze = False ##
output_channels = 80
conv_p = 0.6
h1_p = 0.6
#sim_p = 0.2
#h2_p = 0.6
batch_size = 512 ##
folder = 5
early_stop = 10
LR = 0.001
Gamma = 0.99
num_epochs = 150
version = "v01"
##--------------parameters-------------------##

def train():
    SPACE = "words" ##TODO put parameters here
    OUT_CHANS = 100

    train_df = DataSet.load_train()
    xtr_df, xval_df = train_test_split(train_df, test_size=0.25)
    test_df = DataSet.load_test()
    ### Generate data generator
    train_dg = DataGenerator(data_df=xtr_df,space=SPACE,bucket_num=5,batch_size=5000,is_prefix_pad=True,is_shuffle=True,is_test=False)
    val_dg = DataGenerator(data_df=xval_df,space=SPACE,bucket_num=5,batch_size=5000,is_prefix_pad=True,is_shuffle=False,is_test=False)
    test_dg = DataGenerator(data_df=test_df,space=SPACE,bucket_num=5,batch_size=5000,is_prefix_pad=True,is_shuffle=False,is_test=True)
    ### Must do prepare before using
    train_dg.prepare()
    val_dg.prepare()
    test_dg.prepare()
    ### load word embedding, can use train_df, val_dg or test_dg
    item_embed = train_dg.get_item_embed_tensor(SPACE)
    ### Initialize network
    siamese_cnn = Siamese_CNN(item_embed,is_freeze=True,output_channels=OUT_CHANS)
    ### Initialize model using network
    siamese_model = Model(siamese_cnn)
    criteria = nn.BCEWithLogitsLoss()
    optimizer_ft = optim.Adam(ifilter(lambda p: p.requires_grad, siamese_cnn.parameters()), lr=9e-4, weight_decay=0)##
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.98)
    ### Train
    siamese_model.train(train_dg,val_dg,criteria,optimizer_ft,exp_lr_scheduler,100)
    preds = siamese_model.predict(test_dg).numpy()
    preds = pd.DataFrame({"y_pre":preds})
    preds.to_csv("submission.csv",index=False)

def cv_main():
    kf = KFold(n_splits=folder,shuffle=True,random_state=19920618)
    all_train_df = DataSet.load_train()
    test_df = DataSet.load_test()
    test_dg = DataGenerator(data_df=test_df,space=space,bucket_num=5,batch_size=256,is_prefix_pad=True,is_shuffle=False,is_test=True)
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
        train_dg = DataGenerator(data_df=xtr_df, space=space, bucket_num=5, batch_size=batch_size, is_prefix_pad=True,
                                 is_shuffle=True, is_test=False)
        val_dg = DataGenerator(data_df=xval_df, space=space, bucket_num=5, batch_size=256, is_prefix_pad=True,
                               is_shuffle=False, is_test=False)
        print("prepare train data generator, cv_%s"%i)
        train_dg.prepare()
        print("prepare val data generator, cv_%s" % i)
        val_dg.prepare()
        siamese_lstm = Siamese_CNN(pre_trained_embedding=item_embed,
                                   is_freeze=is_freeze,
                                   output_channels=output_channels,
                                   conv_p = conv_p,
                                   h1_p = h1_p

                                   )
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
    cv_main()
