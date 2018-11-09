
import gzip
import numpy as np
import torch
import pickle
import torch.utils.data as data
from project.data.make_features import create_data

def make_datasets(args):
    baseX, progX, text, labels = create_data(args.max_base, args.max_prog, args.max_before, args.max_after, args.desired_features)
    split = int(args.valid_split*baseX.size(0))

    print(baseX.size(0))

    baseX_valid, baseX_train = baseX[:split], baseX[split:]
    progX_valid, progX_train = progX[:split], progX[split:]
    text_valid, text_train = text[:split], text[split:]
    labels_valid, labels_train = labels[:split], labels[split:]

    trainDataset = TotalData(baseX_train, progX_train, text_train, labels_train)
    validDataset = TotalData(baseX_valid, progX_valid, text_valid, labels_valid)

    return trainDataset, validDataset

class TotalData(data.Dataset):
    def __init__(self, baseX, progX, text, labels):
        self.baseX, self.progX, self.text, self.labels = baseX, progX, text, labels

    def __len__(self):
        return self.baseX.size(0)

    def __getitem__(self, idx):
        return {"baseX": self.baseX[idx], "progX": self.progX[idx], "text": self.text[idx], "labels": self.labels[idx]}
