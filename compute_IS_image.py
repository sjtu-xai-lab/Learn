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
from util.train_util import prepare_model

from and_or_interaction.harsanyi_calculator_img_clean_tau import HarsanyiImg


def compute_harsanyi(model, X, y, save_dir):

    harsanyi = HarsanyiImg(
        model=model,
        device=args.device,
        reward_function=args.reward_function,
        image_size=args.image_size,
        grid_scale=args.grid_scale,
        n_dim=args.n_dim,
        batch_size=args.bs,
        mean=[0,0,0],
        std=[1,1,1],
        seed=args.seed,
        save_dir=save_dir,
        background="baseline" # TODO: "baseline" or "ori", "baseline" represents the rest 52 patches are masked.
    )

    tmp_0 = torch.tensor(0).to(args.device)
    tau = torch.tensor(args.tau).to(args.device)

    for i in range(X.shape[0]):
        name = 'sample_' + str(i)
        data = X[i].unsqueeze(0).to(args.device)
        label = y[i].item()
        fname = '-'.join(["%s"%name, f'label_{label}']) # TODO: you can change the name of the saving folder for a specific sample, can be replaced with your own name format

        # The variable region_idx specifies the set of patches we are going to compute Harsanyi
        chosen_region = np.random.RandomState(seed=i).randint(len(region_idx_rect)) #chosen from the template region idx
        if args.region_type == "rect": # choose from one of the templates
            region_idx = region_idx_rect[chosen_region]
        elif args.region_type == "irreg": # choose from one of the templates
            region_idx = region_idx_irreg[chosen_region]
        elif args.region_type == "random": # randomly choose 12 patches from central 6x6 region
            central_region_idx = np.arange(args.grid_scale**2).reshape(args.grid_scale,args.grid_scale)[1:-1, 1:-1] # 6x6 region
            region_idx = np.random.RandomState(seed=i).choice(central_region_idx.flatten(), size=args.n_dim, replace=False) # random choose 12 patches
            #region_idx = np.random.RandomState(seed=args.seed).choice(central_region_idx.flatten(), size=args.n_dim, replace=False) # fixed random choose 12 patches
        else:
            raise NotImplementedError

        baseline = torch.zeros_like(data) # TODO: you can also use the tau baseline

        mask_pos = (data > 0) # greater than the mean value of the whole dataset
        mask_neg = (data <= 0) # less than the mean value of the whole dataset

        baseline[mask_pos] = torch.maximum(data[mask_pos] - tau, tmp_0)
        baseline[mask_neg] = torch.minimum(data[mask_neg] + tau, tmp_0)

        # This fuction calculates the interaction
        v_N, v_Empty, rewards, Iand = harsanyi.compute_harsanyi(image=data, target=label, baseline_value = baseline,
                                                                     region_idx=region_idx, fname=fname, visual=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--reward_function', type=str, default='logit', choices=['logit', 'output'],
                        help='logit means using log p/(1-p) to compute interaction'
                             'output means using the output score before the softmax layer')
    parser.add_argument('--bs', type=int, default = 512, help='batch size to compute harsanyi')


    parser.add_argument("--dataset", default="cifar10", type=str, choices=['cifar10', 'tiny200'],
                        help="dataset name, currently only support 'imagenet' and 'cifar10' ")
    parser.add_argument('--device', default=1, type=int, help="GPU ID")
    parser.add_argument("--arch", default="our_resnet20", type=str, choices=['our_vgg16', 'our_vgg19', 'our_alexnet', 
    'our_resnet20', 'our_resnet56','our_resnet18', 'our_resnet50'])
    parser.add_argument("--model_path", default="", type=str, help= "path for the pre-trained model")
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--batch-size', default=500, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--data_num', default = 10, type=int, metavar='N',
                        help='number of data samples for computation')
    parser.add_argument('--train_epoch', default=200, type=int, metavar='N')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--tau', default=2, type=float, help="tau for setting baseline")

    
    parser.add_argument('--grid_scale', default=8, type=int, metavar='N', help="segment the image to 8x8 patches")
    parser.add_argument('--n_dim', default=12, type=int, metavar='N', help = "number of grids to compute harsanyi, here we use 12")
    parser.add_argument('--region_type', type=str, default="random", choices=["rect","irreg","random"]) # rectangular region or irregular region or randomly choose patches to compute Harsanyi
    args = parser.parse_args()

    set_seed(args.seed)


    # manually define the region_idx templates for rectangular and irregular regions
    region_idx_rect = [
    [9,10,11,12, 17,18,19,20, 25,26,27,28],
    [19,20,21,22, 27,28,29,30, 35,36,37,38],
    [34,35,36,37, 42,43,44,45, 50,51,52,53],
    # [11,12,13,14, 19,20,21,22, 27,28,29,30],
    ]

    region_idx_irreg = [
    [10,12, 17,18,19,20,21, 25,27,28,29, 37], # test16
    [18,20,22, 26,27,28,29,30, 36,37, 43,44], # test14 shift up
    [27, 34,35,36,38, 42,43,44,45,46, 51,53], # test2 shift left
    # [11,13, 18,19,20,21,22, 26,28,29,30, 38],
    ]

    category_all = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # choose the category
    epoch_all = np.array([100, 199]) # choose the certain epoch

    
    CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR10_STD = [0.2471, 0.2435, 0.2616]

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    save_path = "/data1/results/IS_" + args.dataset \
        + '_' + str(args.region_type) + '_' + str(args.arch) + '_' + str(args.data_num)+ '_ada_' + str(args.tau)

    makedirs(save_path)
    os.system(f"cp {os.path.abspath(__file__)} {save_path}/compute_IS_image.backup")

    model_path_all = []
    
    for i in epoch_all:
        model_path_temp = args.model_path+str(i)+'.pth'
        model_path_all.append(model_path_temp)

    print(model_path_all)

    model = prepare_model(args)

    # prepare data

    if args.dataset == "cifar10":

        args.image_size = 32
        normalize = transforms.Normalize(mean=CIFAR10_MEAN,
                                         std=CIFAR10_STD)
        train_transform_list = []
        train_transform_list.append(transforms.ToTensor())
        train_transform_list.append(normalize)
        print("train transform list: ", train_transform_list)
        train_transform = transforms.Compose(train_transform_list)
        train_set = datasets.CIFAR10(root='/data1/image', train=True, transform=train_transform, download=True) # dataset path
        
    elif args.dataset == "tiny200":
        
        args.image_size = 224
        num_class = int(args.dataset[len("tiny"):])
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        train_transform_list = []
        train_transform_list.append(transforms.Resize((224, 224)))
        train_transform_list.append(transforms.ToTensor())
        train_transform_list.append(normalize)
        train_transform = transforms.Compose(train_transform_list)
        train_set = datasets.ImageFolder(root='/data1/image/tiny-imagenet/train/', transform=train_transform) # dataset path

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle=False, num_workers=args.workers)

    
    for category in category_all:

        # sample data

        for i, (input, target) in enumerate(train_loader):
            label_index = torch.where(target==category)
            sample_data = input[label_index]
            sample_label = target[label_index]

            if sample_data.shape[0] > args.data_num:
                sample_data = sample_data[0:args.data_num,:,:,:]
                sample_label = sample_label[0:args.data_num]
                break
            else:
                print('data number of sampling is not enough')
     
        X = sample_data
        y = sample_label


        for stat, model_path in enumerate(model_path_all):

            print('current model is', epoch_all[stat])
            save_path_temp = save_path + '/model_epoch_' + str(epoch_all[stat])
            makedirs(save_path_temp)

            checkpoint = torch.load(model_path, map_location='cuda')
            model.load_state_dict(checkpoint['state_dict'])
            model.to(args.device)
            model.eval()
            compute_harsanyi(model, X, y, save_path_temp)
