import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import tqdm
import datetime
import pdb

class TextNet(nn.Module):
    def __init__(self, args):
        super(FeatureNet, self).__init__()


        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)
        self.output = nn.Linear(args.max_prog + args.max_base, 4)
        self.softmax = nn.Softmax()

        self.val_acc=0
        self.args = args

    def forward(self, x):

        first_out = self.output(x)
        nonlinear = self.relu(first_out)
        dropout = self.dropout(nonlinear)
        return self.softmax(dropout)

    def set_accuracy(self, acc):
        self.val_acc=acc

    def get_accuracy(self):
        return self.val_acc

    def get_args(self):
        return self.args
        
class FeatureNet(nn.Module):
    def __init__(self, args, concat_func):
        super(FeatureNet, self).__init__()

        self.shared_layer = nn.Linear(args.max_before+args.max_after+len(args.desired_features), args.mid_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)
        self.output = nn.Linear(args.mid_dim*2, 4, bias=False)
        self.softmax = nn.Softmax()
        self.concat_func = concat_func
        self.val_acc=0
        self.args = args

    def forward(self, x, y):
        first_base = self.relu(self.shared_layer(x))
        second_base = self.concat_func(first_base, 1)

        first_progress = self.relu(self.shared_layer(y))
        second_progress = self.concat_func(first_progress, 1)

        overall_train = torch.cat((second_base, second_progress), dim=-1)

        overall_out  =  self.output(overall_train)

        return self.softmax(overall_out)

    def set_accuracy(self, acc):
        self.val_acc=acc

    def get_accuracy(self):
        return self.val_acc

    def get_args(self):
        return self.args
class CombinedNet(nn.Module):

    def __init__(self, args, concat_func):
        super(CombinedNet, self).__init__()

        self.shared_layer = nn.Linear(args.max_before+args.max_after+len(args.desired_features), args.mid_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)
        self.output = nn.Linear(args.mid_dim*2 + args.max_prog + args.max_base, 4, bias=False)
        self.softmax = nn.Softmax()
        self.concat_func = concat_func
        self.val_acc=0
        self.args = args

    def forward(self, x, y, z):
        first_base = self.relu(self.shared_layer(x))
        second_base = self.concat_func(first_base, 1)

        first_progress = self.relu(self.shared_layer(y))
        second_progress = self.concat_func(first_progress, 1)

        overall_train = torch.cat((second_base, second_progress, z), dim=-1)

        overall_out  =  self.output(overall_train)

        return self.softmax(overall_out)

    def set_accuracy(self, acc):
        self.val_acc=acc

    def get_accuracy(self):
        return self.val_acc

    def get_args(self):
        return self.args
