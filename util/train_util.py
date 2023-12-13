import os
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models.resnet_cifar import our_resnet20, our_resnet56
from models.resnet import our_resnet18, our_resnet50
from models.vgg import our_vgg16, our_vgg19
from models.alexnet import our_alexnet
from models.mlp import mlp5, mlp8, mlp3

from util.utils import LogWriter, IMAGENET_MEAN, IMAGENET_STD, CIFAR10_MEAN, CIFAR10_STD, CELEBA_MEAN, CELEBA_STD
import socket
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def log_hparam_and_backup_code(args, file_path):
    file_name = os.path.basename(file_path)
    logfile = LogWriter(os.path.join(args.result_path, "hparam.txt"))
    for k, v in args.__dict__.items():
        logfile.cprint(f"{k} : {v}")
    logfile.cprint("Numpy: {}".format(np.__version__))
    logfile.cprint("Pytorch: {}".format(torch.__version__))
    logfile.cprint("torchvision: {}".format(torchvision.__version__))
    logfile.cprint("Cuda: {}".format(torch.version.cuda))
    logfile.cprint("hostname: {}".format(socket.gethostname()))
    logfile.close()

    os.system(f'cp {file_path} {args.result_path}/{file_name}.backup')

def prepare_model(args):
    # todo: add more models
    if args.arch == "our_vgg16":
        model = our_vgg16(dataset=args.dataset)
    elif args.arch == "our_vgg19":
        model = our_vgg19(dataset=args.dataset)
    elif args.arch == "our_alexnet":
        model = our_alexnet(dataset=args.dataset)

    elif args.arch == "our_resnet20":
        model = our_resnet20(dataset=args.dataset)
    elif args.arch == "our_resnet56":
        model = our_resnet56(dataset=args.dataset)

    elif args.arch == "our_resnet18":
        model = our_resnet18(dataset=args.dataset)
    elif args.arch == "our_resnet50":
        model = our_resnet50(dataset=args.dataset)

    elif args.arch == "mlp5":
        model = mlp5(dataset=args.dataset)
    elif args.arch == "mlp8":
        model = mlp8(dataset=args.dataset)
    elif args.arch == "mlp3":
        model = mlp3(dataset=args.dataset)
    else:
        raise Exception("Model not implemented")
    model.to(args.device)
    print(str(model))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            #checkpoint = torch.load(args.resume, map_location=args.device)
            checkpoint = torch.load(args.resume, map_location='cuda')
            # args.start_epoch = checkpoint['epoch']
            if 'best_prec1' in checkpoint:
                best_prec1 = checkpoint['best_prec1']
                print("best acc", best_prec1)
            if not ('state_dict' in checkpoint):
                sd = checkpoint
            else:
                sd = checkpoint['state_dict']
            # load with models trained on a single gpu or multiple gpus
            if 'module.' in list(sd.keys())[0]:
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            print("=> loaded checkpoint '{}'"
                  .format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    return model

def prepare_dataset(args):
    # todo: add more datasets
    if args.dataset == "cifar10":
        normalize = transforms.Normalize(mean=CIFAR10_MEAN,
                                         std=CIFAR10_STD)

        train_transform_list = []

        if args.horizontal_flip:
            train_transform_list.append(transforms.RandomHorizontalFlip())
        if args.random_crop:
            train_transform_list.append(transforms.RandomCrop(32, 4))

        train_transform_list.append(transforms.ToTensor())
        train_transform_list.append(normalize)
        print("train transform list: ", train_transform_list)
        train_transform = transforms.Compose(train_transform_list)

        train_set = datasets.CIFAR10(root='/data1/image', train=True, transform=train_transform, download=True)
        val_set = datasets.CIFAR10(root='/data1/image', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))

    elif args.dataset == "tiny50" or args.dataset == "tiny10" or args.dataset == 'tiny200':
        num_class = int(args.dataset[len("tiny"):])
        normalize = transforms.Normalize(mean=IMAGENET_MEAN,
                                         std=IMAGENET_STD)
        train_transform_list = []

        if args.random_crop:
            train_transform_list.append(transforms.Resize((256, 256)))
            train_transform_list.append(transforms.RandomCrop(224))
        else:
            train_transform_list.append(transforms.Resize((224, 224)))

        if args.horizontal_flip:
            train_transform_list.append(transforms.RandomHorizontalFlip())

        train_transform_list.append(transforms.ToTensor())
        train_transform_list.append(normalize)
        print("train transform list: ", train_transform_list)
        train_transform = transforms.Compose(train_transform_list)

        train_set = datasets.ImageFolder(root='/data1/image/tiny-imagenet-%d/train/'%num_class, transform=train_transform)
        val_set = datasets.ImageFolder(root='/data1/image/tiny-imagenet-%d/val_split/'%num_class, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]))

    elif args.dataset == "commercial" or args.dataset == "census":
        data_folder = f"/data1/tabular/{args.dataset}"
        X_train = np.load(os.path.join(data_folder, "X_train.npy"))
        y_train = np.load(os.path.join(data_folder, "y_train.npy"))
        X_test = np.load(os.path.join(data_folder, "X_test.npy"))
        y_test = np.load(os.path.join(data_folder, "y_test.npy"))

        X_train = torch.from_numpy(X_train).float() # do not add .to(args.device) here
        y_train = torch.from_numpy(y_train).long()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).long()

        # create dataloader
        train_set = torch.utils.data.TensorDataset(X_train, y_train)
        val_set = torch.utils.data.TensorDataset(X_test, y_test) # actually this is test set, for compatability we rename it to val_set
    else:
        raise Exception("dataset [%s] not implemented" % args.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)
    return train_loader, val_loader

def get_tabular_dataset_bound(args):
    if args.dataset == "commercial" or args.dataset == "census":
        data_folder = f"datasets/{args.dataset}"
        X_train = np.load(os.path.join(data_folder, "X_train.npy"))
        X_test = np.load(os.path.join(data_folder, "X_test.npy"))

        X_train = torch.from_numpy(X_train).float()  # do not add .to(args.device) here
        X_test = torch.from_numpy(X_test).float()

        bound_max_train = torch.max(X_train, dim=0)[0].to(args.device)
        bound_min_train = torch.min(X_train, dim=0)[0].to(args.device)
        bound_max_test = torch.max(X_test, dim=0)[0].to(args.device)
        bound_min_test = torch.min(X_test, dim=0)[0].to(args.device)
    else:
        raise Exception("dataset [%s] not implemented" % args.dataset)

    return bound_max_train, bound_min_train, bound_max_test, bound_min_test


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res