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


class Net(nn.Module):

    def __init__(self, architecture, feat_dim, text_dim):
        super(Net, self).__init__()
        self.architecture = architecture

        self.shared_layer = nn.Linear(feat_dim, 100)
        self.relu = nn.Relu()
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(100*2 + text_dim, 4)
        self.softmax = nn.Softmax()

    def forward(self, x, y, z, concatenation_function=torch.sum):
        first_base = self.relu(self.shared_layer(x))
        second_base = concatenation_function(first_base, axis=1)

        first_progress = self.relu(self.shared_layer(y))
        second_progress = concatenation_function(first_progress, axis=1)

        overall_train = torch.cat(second_base, second_progress, z)

        overall_out  =  self.output(overall_train)

        return self.softmax(overall_out)
