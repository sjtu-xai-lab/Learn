import argparse
import math
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader
matplotlib.use("Agg")
from tools.library import *
from tqdm import tqdm
import time
import sys
from typing import Iterable, Tuple, Union
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, permutations
from models.mlp import mlp5, mlp8

from and_or_interaction.harsanyi_calculator_tabular_clean import HarsanyiTabular


def compute_harsanyi_new(model, X, y, save_dir):

    harsanyi = HarsanyiTabular(
        model=model,
        device=args.device,
        reward_function=args.reward_function,
        n_dim=args.n_dim,
        seed=args.seed,
        save_dir=save_dir
    )

    for i in range(X.shape[0]):
        name = 'sample_' + str(i)
        data = X[i].unsqueeze(0)
        label = y[i].item()
        fname = '-'.join(["%s"%name, f'label_{label}']) # TODO: you can change the name of the saving folder for a specific sample, can be replaced with your own name format
        baseline = torch.zeros_like(data) # TODO: you can also use the tau baseline

        mask_pos = (data > 0) # greater than the mean value of the whole dataset
        mask_neg = (data<= 0) # less than the mean value of the whole dataset

        tmp_0 = torch.tensor(0).to(args.device)


        baseline[mask_pos] = torch.maximum(data[mask_pos] - args.tau, tmp_0)
        baseline[mask_neg] = torch.minimum(data[mask_neg] + args.tau, tmp_0)

        # This fuction calculates the and interaction
        v_N, v_Empty, rewards, Iand = harsanyi.compute_harsanyi(data=data, target=label,
                                                                     baseline=baseline, fname=fname)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--reward_function', type=str, default='logit', choices=['logit', 'output'],
                        help='logit means using log p/(1-p) to compute interaction'
                             'output means using the output score before the softmax layer')

    parser.add_argument("--dataset", default="commercial", type=str, choices=['census','commercial', 'generate'],
                        help="dataset name, currently only support 'imagenet' and 'cifar10' ")
    parser.add_argument('--arch', default="mlp5",type=str) # TODO: use your own model
    parser.add_argument('--device', default=1, type=int, help="GPU ID")
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--tau', default=1, type=float, help="tau for setting baseline")
    parser.add_argument('--data_num', default=50, type=int, help="number of samples")
    parser.add_argument("--model_path", default="", type=str, help= "path for the pre-trained model")
    
    args = parser.parse_args()
    set_seed(args.seed)

    save_path = "/data1/results/IS_" + args.dataset + '_' + args.arch + '_' + str(args.data_num)+ '_ada_' + str(args.tau)
    makedirs(save_path)
    os.system(f"cp {os.path.abspath(__file__)} {save_path}/compute_IS.backup")

    args.n_dim = 12 if args.dataset == "census" else 10 # 12 for census, 10 for commercial


    category_all = [0, 1]
    epoch_all = np.array([100, 199])

    model = mlp5(dataset=args.dataset)

    model_path_all = []
    
    for i in epoch_all:
        model_path_temp = args.model_path+str(i)+'.pth'
        model_path_all.append(model_path_temp)

    print(model_path_all)

    # load data

    if args.dataset == 'census':
        X_0 = np.load('/data1/calculate_IS_tabular/tabular/census/X_train_0.npy') 
        X_1 = np.load('/data1/calculate_IS_tabular/tabular/census/X_train_1.npy')
        y_0 = np.load('/data1/calculate_IS_tabular/tabular/census/y_train_0.npy')
        y_1 = np.load('/data1/calculate_IS_tabular/tabular/census/y_train_1.npy') 

    elif args.dataset == 'commercial':
        X_0 = np.load('/data1/calculate_IS_tabular/tabular/commercial/X_train_0.npy')
        X_1 = np.load('/data1/calculate_IS_tabular/tabular/commercial/X_train_1.npy')
        y_0 = np.load('/data1/calculate_IS_tabular/tabular/commercial/y_train_0.npy')
        y_1 = np.load('/data1/calculate_IS_tabular/tabular/commercial/y_train_1.npy')


    for stat, model_path in enumerate(model_path_all):

        print('current model is', epoch_all[stat])
        save_path_temp = save_path + '/model_epoch_' + str(epoch_all[stat])
        makedirs(save_path_temp)

        checkpoint = torch.load(model_path, map_location='cuda')
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        for category in category_all:

            if category == 0:
                X = X_0[0:args.data_num,:]
                y = y_0[0:args.data_num]

            else:
                X = X_1[0:args.data_num,:]
                y = y_1[0:args.data_num]

            X = torch.from_numpy(X).float().to(args.device)
            y = torch.from_numpy(y).long().to(args.device)

            compute_harsanyi_new(model, X, y, save_path_temp)
