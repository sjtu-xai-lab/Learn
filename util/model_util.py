import os
import torch
import torch.nn as nn
import torchvision

from models.resnet_cifar import our_resnet20, our_resnet56
from models.resnet import our_resnet18, our_resnet50
from models.vgg import our_vgg16, our_vgg19
from models.alexnet import our_alexnet
from models.mlp import mlp5, mlp8


def load_checkpoint(args, checkpoint_path: str, model: nn.Module, optimizer: torch.optim.Optimizer=None) -> None:
    """
    Input
        args: args
        checkpoint_path: str, path of saved model parameters
        model: nn.Module
        optimizer: torch.optim.Optimizer
    Return:
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'File doesn\'t exists {checkpoint_path}')
    print(f'=> loading checkpoint "{checkpoint_path}"')
    checkpoint = torch.load(checkpoint_path, map_location=args.device)

    # Makes us able to load models saved with legacy versions
    # state_dict_key = 'model'
    if not ('state_dict' in checkpoint):
        sd = checkpoint
    else:
        sd = checkpoint['state_dict']

    # load with models trained on a single gpu or multiple gpus
    if 'module.' in list(sd.keys())[0]:
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)

    print(f'=> loaded checkpoint "{checkpoint_path}"')
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])


# ----- imagenet pretrained models ------

def get_alexnet(args, load_model=True) -> nn.Module:
    model = torchvision.models.alexnet()
    if load_model:
        load_checkpoint(args, args.checkpoint_path, model)
    else:
        print("use model at initialization")
    return model

def get_resnet18(args, load_model=True) -> nn.Module:
    model = torchvision.models.resnet18()
    if load_model:
        load_checkpoint(args, args.checkpoint_path, model)
    else:
        print("use model at initialization")
    return model

def get_resnet50(args, load_model=True) -> nn.Module:
    model = torchvision.models.resnet50()
    if load_model:
        load_checkpoint(args, args.checkpoint_path, model)
    else:
        print("use model at initialization")
    return model

def get_vgg16(args, load_model=True) -> nn.Module:
    model = torchvision.models.vgg16()
    if load_model:
        load_checkpoint(args, args.checkpoint_path, model)
    else:
        print("use model at initialization")
    return model


# ------- our models ---------

def get_our_vgg16(args, load_model=True) -> nn.Module:
    model = our_vgg16(dataset=args.dataset) # we do not use dropout when testing
    if load_model:
        load_checkpoint(args, args.checkpoint_path, model)
    else:
        print("use model at initialization")
    return model

def get_our_vgg19(args, load_model=True) -> nn.Module:
    model = our_vgg19(dataset=args.dataset) # we do not use dropout when testing
    if load_model:
        load_checkpoint(args, args.checkpoint_path, model)
    else:
        print("use model at initialization")
    return model

def get_our_alexnet(args, load_model=True) -> nn.Module:
    model = our_alexnet(dataset=args.dataset) # we do not use dropout when testing
    if load_model:
        load_checkpoint(args, args.checkpoint_path, model)
    else:
        print("use model at initialization")
    return model

def get_our_resnet20(args, load_model=True) -> nn.Module:
    model = our_resnet20(dataset=args.dataset) # we do not use dropout when testing
    if load_model:
        load_checkpoint(args, args.checkpoint_path, model)
    else:
        print("use model at initialization")
    return model

def get_our_resnet56(args, load_model=True) -> nn.Module:
    model = our_resnet56(dataset=args.dataset) # we do not use dropout when testing
    if load_model:
        load_checkpoint(args, args.checkpoint_path, model)
    else:
        print("use model at initialization")
    return model

def get_our_resnet18(args, load_model=True) -> nn.Module:
    model = our_resnet18(dataset=args.dataset) # we do not use dropout when testing
    if load_model:
        load_checkpoint(args, args.checkpoint_path, model)
    else:
        print("use model at initialization")
    return model

def get_our_resnet50(args, load_model=True) -> nn.Module:
    model = our_resnet50(dataset=args.dataset) # we do not use dropout when testing
    if load_model:
        load_checkpoint(args, args.checkpoint_path, model)
    else:
        print("use model at initialization")
    return model


# model for tabular data

def get_mlp5(args, load_model=True) -> nn.Module:
    model = mlp5(dataset=args.dataset)
    if load_model:
        load_checkpoint(args, args.checkpoint_path, model)
    else:
        print("use model at initialization")
    return model

def get_mlp8(args, load_model=True) -> nn.Module:
    model = mlp8(dataset=args.dataset)
    if load_model:
        load_checkpoint(args, args.checkpoint_path, model)
    else:
        print("use model at initialization")
    return model

def get_model(args) -> nn.Module: # todo: add more models
    """ get model and load parameters if needed
    Input:
        args: args
            if args.checkpoint_path is "None", then do not load model parameters
    Return:
        some model: nn.Module, model to be evaluated
    """
    torch.hub.set_dir(args.pretrained_models_dirname)
    if args.checkpoint_path == "None":
        load_model = False
    else:
        load_model = True

    # ----- imagenet pretrained models ------
    if args.arch == 'alexnet':
        assert args.dataset == "imagenet"
        print("imagenet pretrained alexnet")
        return get_alexnet(args, load_model=load_model)
    elif args.arch == 'resnet18':
        assert args.dataset == "imagenet"
        print("imagenet pretrained resnet18")
        return get_resnet18(args, load_model=load_model)
    elif args.arch == 'resnet50':
        assert args.dataset == "imagenet"
        print("imagenet pretrained resnet50")
        return get_resnet50(args, load_model=load_model)
    elif args.arch == 'vgg16':
        assert args.dataset == "imagenet"
        print("imagenet pretrained vgg16")
        return get_vgg16(args, load_model=load_model)

    # ----- our models ------
    elif 'our_vgg16' in args.arch:
        print("use our vgg16")
        return get_our_vgg16(args, load_model=load_model)
    elif 'our_vgg19' in args.arch:
        print("use our vgg19")
        return get_our_vgg19(args, load_model=load_model)
    elif 'our_alexnet' in args.arch:
        print("use our alexnet")
        return get_our_alexnet(args, load_model=load_model)

    elif 'our_resnet20' in args.arch:
        print("use our resnet20")
        return get_our_resnet20(args, load_model=load_model)
    elif 'our_resnet56' in args.arch:
        print("use our resnet56")
        return get_our_resnet56(args, load_model=load_model)

    elif 'our_resnet18' in args.arch:
        print("use our resnet18")
        return get_our_resnet18(args, load_model=load_model)
    elif 'our_resnet50' in args.arch:
        print("use our resnet50")
        return get_our_resnet50(args, load_model=load_model)


    # tabular models
    elif "mlp5" in args.arch:
        print("use mlp5")
        return get_mlp5(args, load_model=load_model)
    elif "mlp8" in args.arch:
        print("use mlp8")
        return get_mlp8(args, load_model=load_model)
    else:
        raise Exception(f"model [{args.arch}] not implemented. Error in get_model.")

