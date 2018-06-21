# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch import nn as nn, autograd
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Siamese_CNN(nn.Module):
    def __init__(self,list_filter_size,if_tune_embedding,pre_trained_embedding):
        super(Siamese_CNN,self).__init__()
        self.list_filter_size = list_filter_size
        self.if_tune_embedding = if_tune_embedding

