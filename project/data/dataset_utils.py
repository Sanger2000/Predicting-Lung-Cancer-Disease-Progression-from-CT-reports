
import gzip
import numpy as np
import torch
import pickle
import torch.utils.data as data
from project.data.make_features import create_data

def make_datasets(args):
    baseX, progX, text, labels, token_ids, segment_ids = create_data(args.max_base, args.max_prog, args.max_before, args.max_after, args.desired_features)
    split = int(args.valid_split*baseX.size(0))

    indices = torch.randperm(baseX.size(0))

    baseX, progX, text, labels, token_ids, segment_ids = baseX[indices], progX[indices], text[indices], labels[indices], token_ids[indices], segment_ids[indices]

    baseX_valid, baseX_train = baseX[:split], baseX[split:]
    progX_valid, progX_train = progX[:split], progX[split:]
    text_valid, text_train = text[:split], text[split:]
    labels_valid, labels_train = labels[:split], labels[split:]
    token_valid, token_train = token_ids[:split], token_ids[split:]
    segment_valid, segment_train = segment_ids[:split], segment_ids[split:]

    trainDataset = TotalData(baseX_train, progX_train, text_train, labels_train, token_train, segment_train)
    validDataset = TotalData(baseX_valid, progX_valid, text_valid, labels_valid, token_valid, segment_valid)

    return trainDataset, validDataset

class TotalData(data.Dataset):
    def __init__(self, baseX, progX, text, labels, token_ids, segment_ids):
        self.baseX, self.progX, self.text, self.labels, self.token_ids, self.segment_ids = baseX, progX, text, labels, token_ids, segment_ids

    def __len__(self):
        return self.baseX.size(0)

    def __getitem__(self, idx):
        return {"baseX": self.baseX[idx], "progX": self.progX[idx], "text": self.text[idx], "labels": self.labels[idx], "token_ids": self.token_ids[idx], "segment_ids": self.segment_ids[idx]}
