import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import socket

# from common import config
from util.model_util import get_model

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2471, 0.2435, 0.2616] # rqh 0909 revise
# CIFAR10_STD = [0.2023, 0.1994, 0.2010]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# todo: this is not the true mean and std on CelebA
CELEBA_MEAN = [0.5, 0.5, 0.5]
CELEBA_STD = [0.5, 0.5, 0.5]

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def clamp(x: int, min: int, max: int) -> int:
    if x < min:
        return min
    elif x > max:
        return max
    else:
        return x


def seed_torch(seed) -> None:
    """ set random seed for all related packages
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'backends'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # rqh 0612 add, to ensure further reproducibility

def get_dataset_util(args, transform: torchvision.transforms.Compose, train: bool):
    """ get dataset
    Input:
        args:
        transform: torchvision.transforms.Compose, transform for the image; tabular data do not need transform
        train: bool, only valid when dataset is NOT ImageNet.
            If train=False, use the validation set. If train=True, use the training set.
            By default we will use the training set. When evaluating on ImageNet, we have to use the validation set.
    Return:
        some dataset: Dataset,
    """
    # todo: add more datasets
    if args.dataset == "imagenet":
        root = os.path.join(args.prefix, args.datasets_dirname, args.dataset_dirname)
        image_root = os.path.join(root, 'val') # we only have validation set for ImageNet
        return ImageNet_selected(args, root, image_root, transform)
    elif args.dataset == "cifar10":
        root = os.path.join(args.prefix, args.datasets_dirname)
        return CIFAR10_selected(args, root, transform, train)
    elif args.dataset == "tiny10" or args.dataset == "tiny50":
        root = os.path.join(args.prefix, args.datasets_dirname, args.dataset_dirname)
        partition_name = "train" if train else "val_split"
        image_root = os.path.join(root, partition_name)
        print("img root", image_root)
        return TinyImageNet_selected(args, root, image_root, transform)
    elif "celeba" in args.dataset:
        root = os.path.join(args.prefix, args.datasets_dirname)
        return CelebA_selected(args, root, transform, train)

    # tabular data
    elif args.dataset == "census" or args.dataset == "commercial":
        root = os.path.join(args.prefix, args.datasets_dirname, args.dataset_dirname)
        return Tabular_selected(args, root, train)
    else:
        raise Exception(f"dataset [{args.dataset}] not implemented. Error in get_dataset_util")

def prepare(args, train:bool) -> Tuple[nn.Module, DataLoader]:
    """ prepare models and dataloader for the computation of multi-order interaction
    Input:
        args: args
        train: bool, only valid when dataset is NOT ImageNet.
            If train=False, use the validation set. If train=True, use the training set.
            By default we will use the training set. When evaluating on ImageNet, we have to use the validation set.
    Return:
        model: nn.Module, model to be evaluated
        dataloader: DataLoader, dataloader of the training/validation set
    """
    # todo: add more datasets
    if args.dataset == "imagenet":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    elif args.dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    elif "tiny" in args.dataset or "celeba" in args.dataset:
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    # tabular data
    elif args.dataset == "census" or args.dataset == "commercial":
        transform = transforms.Compose([transforms.ToTensor()]) # this is actually useless, tabular data do not need transform
    else:
        raise Exception(f"Dataset [{args.dataset}] not implemented. Error in prepareing dataloader")

    dataset = get_dataset_util(args, transform, train)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = get_model(args)
    return model, dataloader

def normalize(args, x: torch.Tensor):
    """ normalize the image before feeding it into the model
    Input:
        args: args
        x : (N,3,H,W) tensor, original image
    Return:
        (x - mean) / std (tensor): (N,3,H,W) tensor, normalized image
    """
    # todo: add more datasets
    if args.dataset == "census" or args.dataset == "commercial": # tabular data already normalized in preprocessing
        return x
    else: # image data should be normalized
        if args.dataset == "imagenet" or "tiny" in args.dataset:
            mean_list, std_list = IMAGENET_MEAN, IMAGENET_STD
        elif args.dataset == "cifar10":
            mean_list, std_list = CIFAR10_MEAN, CIFAR10_STD
        elif "celeba" in args.dataset:
            mean_list, std_list = CELEBA_MEAN, CELEBA_STD
        else:
            raise Exception("Dataset not implemented")
        mean = torch.tensor(mean_list).view(3, 1, 1).expand(x.shape[0], 3, x.shape[2], x.shape[2]).to(x.device)
        std = torch.tensor(std_list).view(3, 1, 1).expand(x.shape[0], 3, x.shape[2], x.shape[2]).to(x.device)
        return (x - mean) / std

class LogWriter():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def log_args_and_backup_code(args, file_path):
    file_name = os.path.basename(file_path)
    logfile = LogWriter(os.path.join(args.output_dir, f"args_{file_name.split('.')[0]}.txt"))
    for k, v in args.__dict__.items():
        logfile.cprint(f"{k} : {v}")
    logfile.cprint("Numpy: {}".format(np.__version__))
    logfile.cprint("Pytorch: {}".format(torch.__version__))
    logfile.cprint("torchvision: {}".format(torchvision.__version__))
    logfile.cprint("Cuda: {}".format(torch.version.cuda))
    logfile.cprint("hostname: {}".format(socket.gethostname()))
    logfile.close()

    os.system(f'cp {file_path} {args.output_dir}/{file_name}.backup')

def get_reward(args, logits, label):
    """ given logits, calculate reward score for interaction computation
    Input:
        args: args.softmax_type determines which type of score to compute the interaction
            - normal: use log p, p is the probability of the {label} class
            - modified: use log p/(1-p), p is the probability of the {label} class
            - yi: use logits the {label} class
        logits: (N,num_class) tensor, a batch of logits before the softmax layer
        label: (1,) tensor, ground truth label
    Return:
        v: (N,) tensor, reward score
    """
    if args.softmax_type == "normal": # log p
        v = F.log_softmax(logits, dim=1)[:, label[0]]
    elif args.softmax_type == "modified": # log p/(1-p)
        v = logits[:, label[0]] - torch.logsumexp(logits[:, np.arange(args.class_number) != label[0].item()],dim=1)
    elif args.softmax_type == "yi": # logits
        v = logits[:, label[0]]
    else:
        raise Exception(f"softmax type [{args.softmax_type}] not implemented")
    return v

def denormalize_img(args, img):
    if "tiny" in args.dataset or args.dataset == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif args.dataset == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    else:
        raise Exception(f"dataset [{args.dataset}] is not supported")
    img_denormalized = torch.zeros_like(img)
    for channel_id in range(3):
        img_denormalized[:, channel_id, :, :] = img[:, channel_id, :, :] * std[channel_id]
        img_denormalized[:, channel_id, :, :] = img_denormalized[:, channel_id, :, :] + mean[channel_id]
    return img_denormalized