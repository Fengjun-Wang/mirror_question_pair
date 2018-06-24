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
import matplotlib.pyplot as plt

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


    def forward(self, input):
        q1,q2  = torch.chunk(input, 2, dim=1) ## Split the question pairs
        q1_embed = self.nn_Embedding(q1) ##NxLxC
        q2_embed = self.nn_Embedding(q2) ##NxLxC
        #batch_size = input.size(0)

        #self.init_hidden(batch_size) ##Initialize h0 and c0
        q1_embed = self.input_dropout(q1_embed)
        q2_embed = self.input_dropout(q2_embed)
        q1_out,q1_hidden = self.lstm(q1_embed)
        q1h,q1c = q1_hidden
        #self.init_hidden(batch_size)
        q2_out,q2_hidden = self.lstm(q2_embed)
        q2h,q2c = q2_hidden

        if self.bidirectional:
            q1_embed = torch.cat((q1h[-2],q1h[-1]),dim=1)
            q2_embed = torch.cat((q2h[-2],q2h[-1]),dim=1)
        else:
            q1_embed = q1h[-1]
            q2_embed = q2h[-1]

        q_pair_embed = torch.cat((q1_embed,q2_embed),dim=1)
        h1 = self.linear1_dropout(F.relu(self.linear1(q_pair_embed)))
        out = self.linear2(h1)
        return out





class Model(object):
    def __init__(self,model):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def train(self,name,train_dg,valid_dg,criterion, optimizer, scheduler, num_epochs):
        since = time.time()
        criterion = criterion.to(self.device)
        dataset_sizes = {"train":len(train_dg),"val":len(valid_dg)}
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        best_loss = sys.maxsize
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
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

                ##Record the train and validation loss and accuracy for analysis
                if phase=="train":
                    train_loss.append(epoch_loss)
                    train_acc.append(epoch_acc)
                else:
                    val_loss.append(epoch_loss)
                    val_acc.append(epoch_acc)

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
        torch.save(self.model.state_dict(),"siamese_CNN_best_pars.pth")

        plt.figure(figsize=(20, 8))
        l1, = plt.plot(train_loss, "r-")
        l2, = plt.plot(val_loss, "g-")
        plt.legend([l1, l2], ["train_loss", "valid_loss"])
        plt.grid()
        plt.savefig(name+"_loss_trend.jpg")
        plt.figure(figsize=(20, 8))
        l1, = plt.plot(train_acc, "r-")
        l2, = plt.plot(val_acc, "g-")
        plt.legend([l1, l2], ["train_acc", "valid_acc"])
        plt.grid()
        plt.savefig(name+"_acc_trend.jpg")
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
    is_freeze = True
    hidden_size = 100
    num_layers = 2
    lstm_dropput_p = 0.6 ##TODO 0.4->0.5->0.6
    lstm_input_dropout = 0.6
    bidirectional = True
    linear_hidden_size = 200
    linear_hid_drop_p = 0.3
    train_name = "v0.2"

    train_df = DataSet.load_train()
    xtr_df, xval_df = train_test_split(train_df, test_size=0.25)
    test_df = DataSet.load_test()
    ### Generate data generator
    train_dg = DataGenerator(data_df=xtr_df,space=space,bucket_num=5,batch_size=512,is_prefix_pad=True,is_shuffle=True,is_test=False)
    val_dg = DataGenerator(data_df=xval_df,space=space,bucket_num=5,batch_size=512,is_prefix_pad=True,is_shuffle=False,is_test=False)
    test_dg = DataGenerator(data_df=test_df,space=space,bucket_num=5,batch_size=512,is_prefix_pad=True,is_shuffle=False,is_test=True)
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
                               lstm_dropout_p=lstm_dropput_p,
                               bidirectional=bidirectional,
                               linear_hid_size=linear_hidden_size,
                               linear_hid_drop_p=linear_hid_drop_p,
                               input_drop_p = lstm_input_dropout)

    ### Initialize model using network
    siamese_model = Model(siamese_lstm)
    criteria = nn.BCEWithLogitsLoss()
    optimizer_ft = optim.Adam(ifilter(lambda p: p.requires_grad, siamese_lstm.parameters()), lr=0.001) ##TODO 0.001
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.99)##TODO 0.99
    ### Train
    siamese_model.train(train_name,train_dg,val_dg,criteria,optimizer_ft,exp_lr_scheduler,150) ##TODO 150
    preds = siamese_model.predict(test_dg).numpy()
    preds = pd.DataFrame({"y_pre":preds})
    preds.to_csv("submission.csv",index=False)


##BCEWithLogitsLoss
#https://discuss.pytorch.org/t/how-to-initiate-parameters-of-layers/1460
if __name__ == "__main__":
    train()
