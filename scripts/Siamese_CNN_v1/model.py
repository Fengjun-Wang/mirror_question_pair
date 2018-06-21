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


