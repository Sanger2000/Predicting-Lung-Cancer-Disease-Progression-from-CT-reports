import argparse
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import project.data.dataset_utils as data_utils
import project.models.model_utils as model_utils
import project.training.train_utils as train_utils
import torch.nn as N
import os
import torch
import datetime
import pickle
import pdb
from pytorch_pretrained_bert import BertForSequenceClassification

parser = argparse.ArgumentParser(description='Lung Cancer Disease Progression Classifier')
# learning
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate [default: 0.001]')
parser.add_argument('--dropout', type=float, default=0.05, help='sets dropout layer [default: 0.4]')
parser.add_argument('--valid_split', type=float, default=0.2, help='sets validation split from data [default: 0.2]')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs for train [default: 30]')
parser.add_argument('--batch_size', type=int, default=10, help='batch size for training [default: 32]')
parser.add_argument('--mid_dim', type=int, default=100, help='middle dimension of feature extraction architecture [default: 100]')
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
parser.add_argument('--save_path', type=str, default="model2.pt", help='Path where to dump model')


args = parser.parse_args()


def test_hyperparamaters(args, model_save_directory="test_models/", dictionary_save_file="data.pkl"):
    MAX_ACC1, MAX_ACC2, MAX_ACC3 = 0, 0, 0
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
    out_dict = {}
    train_data, valid_data = data_utils.make_datasets(args)
    for concat_func in (torch.sum, torch.max, torch.mean, torch.min):
        for lr in (0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01):
            args.lr = lr
            for dropout in (0.05, 0.1, 0.2, 0.3, 0.4):
                args.dropout = dropout
                for mid_dim in (20, 50, 100, 200, 300):
                    args.mid_dim = mid_dim

                    args.save_path=model_save_directory + 'model_concat.pt'
                    model1 = model_utils.CombinedNet(args, concat_func)
                    print(model1)
                    MAX_ACC1, temp_max_acc = train_utils.train_model(train_data, valid_data, model1, args, MAX_ACC1, "combined")
                    if temp_max_acc not in out_dict:
                        out_dict[temp_max_acc] = [(concat_func, lr, dropout, mid_dim, "combined")]
                    else:
                        out_dict[temp_max_acc].append((concat_func, lr, dropout, mid_dim, "combined"))

                    print()

                    args.save_path=model_save_directory + 'model_features.pt'
                    model2 = model_utils.FeatureNet(args, concat_func)
                    print(model2)
                    MAX_ACC2, temp_max_acc = train_utils.train_model(train_data, valid_data, model2, args, MAX_ACC2, "features")
                    if temp_max_acc not in out_dict:
                        out_dict[temp_max_acc] = [(concat_func, lr, dropout, mid_dim, "features")]
                    else:
                        out_dict[temp_max_acc].append((concat_func, lr, dropout, mid_dim, "features"))
                    print()

                    args.save_path=model_save_directory + 'model_text.pt'
                    model3 = model_utils.TextNet(args)
                    print(model3)
                    MAX_ACC3, temp_max_acc = train_utils.train_model(train_data, valid_data, model3, args, MAX_ACC3, "text")
                    if temp_max_acc not in out_dict:
                        out_dict[temp_max_acc] = [(concat_func, lr, dropout, mid_dim, "text")]
                    else:
                        out_dict[temp_max_acc].append((concat_func, lr, dropout, mid_dim, "text"))
                    print()

                    pickle.dump(out_dict, open(dictionary_save_file, "wb"))

def finetune_bert(args, model_save_directory, bert_file_path):

    train_data, valid_data = data_utils.make_datasets(args)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",  num_labels=4)

    args.save_path = model_save_directory + "bert_model.pt"
    ACC, _ = train_utils.train_model(train_data, valid_data, model, args, 0, "bert")

if __name__ == '__main__':
    #test_hyperparamaters(args)
    finetune_bert(args, "models", "BertModels/cased_bert.bin")
