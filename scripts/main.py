import argparse
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import project.data.dataset_utils as data_utils
import project.models.model_utils as model_utils
import project.training.train_utils as train_utils
import os
import torch
import datetime
import pickle
import pdb

parser = argparse.ArgumentParser(description='Lung Cancer Disease Progression Classifier')
# learning
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('--dropout', type=float, default=0.4, help='sets dropout layer [default: 0.4]')
parser.add_argument('--valid_split', type=float, default=0.2, help='sets validation split from data [default: 0.2]')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs for train [default: 30]')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training [default: 32]')
parser.add_argument('--max_base', type=int, default=400, help='maximum text features for baseline data [default: 400]')
parser.add_argument('--max_prog', type=int, default=800, help='maximum text features for baseline data [default: 800]')
parser.add_argument('--max_before', type=int, default=600, help='maximum text features for context before volume [default: 600]')
parser.add_argument('--max_after', type=int, default=300, help='maximum text features for context after volume [default: 300]')
parser.add_argument('--desired_features', type=tuple, default=("lens", "organs", "date_dist"), help='enter context features in format - (\"feat_1\", ..., \"feat_n\")')
# device
parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu [default: True]')
parser.add_argument('--train', action='store_true', default=False, help='enable train [default: False]')
# task
parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
parser.add_argument('--save_path', type=str, default="model.pt", help='Path where to dump model')


args = parser.parse_args()

if __name__ == '__main__':
    # update args and print
    MAX_ACC = 0
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
    for valid_split in (0.2, 0.3):
        args.valid_split = valid_split

        train_data, valid_data = data_utils.make_datasets(args)

        for lr in (0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01):
            args.lr=lr
            for batch_size in (10, 16, 20, 24, 30):
                args.batch_size = batch_size
                for dropout in (0.05, 0.1, 0.2, 0.3):
                    args.dropout = dropout

                    model = model_utils.Net(args)
                    print(model)

                    print()

                    MAX_ACC = train_utils.train_model(train_data, valid_data, model, args, MAX_ACC)
