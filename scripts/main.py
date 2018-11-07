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
parser.add_argument('--epochs', type=int, default=30, help='number of epochs for train [default: 30]')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training [default: 32]')
parser.add_argument('--max_base', type=int, default=400, help='maximum text features for baseline data [default: 400]')
parser.add_argument('--max_prog', type=int, default=800, help='maximum text features for baseline data [default: 800]')
parser.add_argument('--max_before', type=int, default=600, help='maximum text features for context before volume [default: 600]')
parser.add_argument('--max_after', type=int, default=300, help='maximum text features for context after volume [default: 300]')
parser.add_argument('--desired_features', type=tuple, default=("lens", "organs", "date_dist"), help='enter context features in format - (\"feat_1\", ..., \"feat_n\")')
# device
parser.add_argument('--cuda', action='store_true', default=True, help='enable the gpu [default: True]')
parser.add_argument('--train', action='store_true', default=False, help='enable train [default: False]')
# task
parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
parser.add_argument('--save_path', type=str, default="model.pt", help='Path where to dump model')


args = parser.parse_args()

if __name__ == '__main__':
    # update args and print

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    train_data = data_utils.TotalData(args)

    # model
    if args.snapshot is None:
        model = model_utils.Net(args)
    else :
        print('\nLoading model from [%s]...' % args.snapshot)
        try:
            model = torch.load(args.snapshot)
        except :
            print("Sorry, This snapshot doesn't exist."); exit()
    print(model)

    print()
    # train
    if args.train :
        train_utils.train_model(train_data, model, args)
