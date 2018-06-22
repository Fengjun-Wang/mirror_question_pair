# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch import nn as nn, autograd
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Siamese_CNN(nn.Module):
    def __init__(self,pre_trained_embedding,is_freeze):
        super(Siamese_CNN,self).__init__()
        self.nn_Embedding = nn.Embedding.from_pretrained(pre_trained_embedding,freeze=is_freeze)
        input_channel_len = pre_trained_embedding.size(1)
        self.conv1d_size2 = nn.Conv1d(in_channels=input_channel_len,
                                      out_channels=100,
                                      kernel_size=2,
                                      stride=1,
                                      padding=1)
        self.conv1d_size3 = nn.Conv1d(input_channel_len=input_channel_len,
                                      out_channels=100,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        self.conv1d_size4 = nn.Conv1d(input_channel_len=input_channel_len,
                                      out_channels=100,
                                      kernel_size=4,
                                      stride=1,
                                      padding=1)
        self.out_hidden1 = nn.Linear(600,300)
        self.out_hidden2 = nn.Linear(300,150)
        self.out_put = nn.Linear(150,1)

    def forward(self, input):
        q1,q2  = torch.chunk(input, 2, dim=1) ## Split the question pairs
        q1_embed = self.nn_Embedding(q1).transpose(1,2) ##NxLxC -> NxCxL
        q2_embed = self.nn_Embedding(q2).transpose(1,2)

        q1_conv1 = F.relu(self.conv1d_size2(q1_embed)) ##NxCxL
        q1_pool1,_ = q1_conv1.min(dim=2) ##NxC
        q1_conv2 = F.relu(self.conv1d_size3(q1_embed)) ##NxCxL
        q1_pool2,_ = q1_conv2.min(dim=2) ##NxC
        q1_conv3 = F.relu(self.conv1d_size4(q1_embed)) ##NxCxL
        q1_pool3,_ = q1_conv3.min(dim=2) ##NxC(100)
        q1_concat = torch.cat((q1_pool1,q1_pool2,q1_pool3),dim=1) ## Nx(c1+c2...)[300]

        q2_conv1 = F.relu(self.conv1d_size2(q2_embed)) ##NxCxL
        q2_pool1,_ = q2_conv1.min(dim=2) ##NxC
        q2_conv2 = F.relu(self.conv1d_size3(q2_embed)) ##NxCxL
        q2_pool2,_ = q2_conv2.min(dim=2) ##NxC
        q2_conv3 = F.relu(self.conv1d_size4(q2_embed)) ##NxCxL
        q2_pool3,_ = q2_conv3.min(dim=2) ##NxC(100)
        q2_concat = torch.cat((q2_pool1,q2_pool2,q2_pool3),dim=1) ## Nx(c1+c2...)[300]

        q_concat = torch.cat((q1_concat,q2_concat)) ##Nx600
        h1 = F.relu(self.out_hidden1(q_concat))
        h2 = F.relu(self.out_hidden2(h1))
        outscore = self.out_put(h2)
        return outscore



##BCEWithLogitsLoss
#https://discuss.pytorch.org/t/how-to-initiate-parameters-of-layers/1460