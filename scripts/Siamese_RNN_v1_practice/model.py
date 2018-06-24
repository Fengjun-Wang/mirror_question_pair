# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
sys.path.append("../")
import torch
import copy
import time
import pandas as pd
import torch.optim as optim
from feature_engineering.util import DataSet
from sklearn.model_selection import train_test_split
from itertools import ifilter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch import nn as nn, autograd
from torch.nn import functional as F
from torch.optim import lr_scheduler
from data import DataGenerator
from torch.autograd import Variable


class Siamese_RNN(nn.Module):
    def __init__(self,pre_trained_embedding,is_freeze, hidden_dims):
        super(Siamese_RNN,self).__init__()
        self.nn_Embedding = nn.Embedding.from_pretrained(pre_trained_embedding,freeze=is_freeze)
        input_channel_len = pre_trained_embedding.size(1) #300
        self.dropout = nn.Dropout(p=0.4) ###
        self.hidden_dims = hidden_dims # hidden cell dimension
        self.lstm_rnn = nn.LSTM(input_size=input_channel_len, hidden_size=self.hidden_dims, num_layers=1)

    def initialize_hidden_plus_cell(self, batch_size):
        """ Re-initializes the hidden state, cell state, and the forget gate bias of the network. """
        zero_hidden = Variable(torch.randn(1, batch_size, self.opt.hidden_dims))
        zero_cell = Variable(torch.randn(1, batch_size, self.opt.hidden_dims))
        return zero_hidden, zero_cell

    def forward(self, batch_size, input_data, hidden, cell):
        """ Performs a forward pass through the network. """
        q1, q2 = torch.chunk(input, 2, dim=1)  ## Split the question pairs
        q1_embed = self.nn_Embedding(q1).transpose(1, 2)  ##NxLxC -> NxCxL
        q2_embed = self.nn_Embedding(q2).transpose(1, 2)
        output = self.nn_Embedding(input_data).view(1, batch_size, -1)
        # output = self.nn_Embedding(input_data).view(1, batch_size, -1)
        for _ in range(self.opt.num_layers):
            output, (hidden, cell) = self.lstm_rnn(output, (hidden, cell))
        return output, hidden, cell

class Model(object):
    def __init__(self,model):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def train(self,train_dg,valid_dg,criterion, optimizer, scheduler, num_epochs):
        since = time.time()
        criterion = criterion.to(self.device)
        dataset_sizes = {"train":len(train_dg),"val":len(valid_dg)}
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        best_loss = sys.maxsize
        for epoch in range(num_epochs):
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
                    #print("batch size:",inputs.size())
                    sample_num += len(inputs)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        preds = F.sigmoid(outputs)>0.5
                        preds = preds.type(torch.cuda.FloatTensor)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    #print(preds.dtype)
                    #print(labels.dtype)
                    running_loss += loss.item() * inputs.size(0) ## criteria size_average is True default
                    running_corrects += torch.sum(preds.data == labels.data) ## running_corrects are tensor dim 0

                epoch_loss = running_loss / dataset_sizes[phase] ##average loss
                epoch_acc = running_corrects.double().item() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f} Num: {}'.format(
                    phase, epoch_loss, epoch_acc,sample_num))

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_loss))
        print('Corresponding val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        torch.save(self.model.state_dict(),"siamese_RNN_best_pars.pth")
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


def train():
    space = "words"
    train_df = DataSet.load_train()[:10]
    xtr_df, xval_df = train_test_split(train_df, test_size=0.25)
    test_df = DataSet.load_test()[:5]

    # print(train_df)
    # print(test_df)

    ### Generate data generator
    train_dg = DataGenerator(data_df=xtr_df,space=space,bucket_num=1,batch_size=5000,is_prefix_pad=True,is_shuffle=True,is_test=False)
    val_dg = DataGenerator(data_df=xval_df,space=space,bucket_num=1,batch_size=5000,is_prefix_pad=True,is_shuffle=False,is_test=False)
    test_dg = DataGenerator(data_df=test_df,space=space,bucket_num=1,batch_size=5000,is_prefix_pad=True,is_shuffle=False,is_test=True)

    ### Must do prepare before using
    train_dg.prepare()
    val_dg.prepare()
    test_dg.prepare()

    ### load word embedding, can use train_df, val_dg or test_dg
    item_embed = train_dg.get_item_embed_tensor(space)
    # print(item_embed)

    ### Initialize network
    siamese_rnn = Siamese_RNN(item_embed,is_freeze=True, hidden_dims=300)
    ### Initialize model using network
    siamese_model = Model(siamese_rnn)
    criteria = nn.BCEWithLogitsLoss()
    optimizer_ft = optim.Adam(ifilter(lambda p: p.requires_grad, siamese_rnn.parameters()), lr=9e-4) ##TODO 0.0008->0.0009
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.98)##TODO 0.95->0.98
    ### Train
    siamese_model.train(train_dg,val_dg,criteria,optimizer_ft,exp_lr_scheduler,100) ##TODO 50->100
    preds = siamese_model.predict(test_dg).numpy()
    preds = pd.DataFrame({"y_pre":preds})
    preds.to_csv("submission.csv",index=False)


##BCEWithLogitsLoss
#https://discuss.pytorch.org/t/how-to-initiate-parameters-of-layers/1460
if __name__ == "__main__":
    train()
