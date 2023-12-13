import os
from typing import List, Tuple

# from common import config
# from dataset import BaseDataset
import torchvision
from torch.utils.data import Dataset, Subset
from io_handler import SampleIoHandler
from PIL import Image
import numpy as np


class ImageNet_selected(Dataset):

    def __init__(self, args, root: str, image_root: str, transform: torchvision.transforms.Compose) -> None:
        super().__init__()
        self.root = root
        self.image_root = image_root
        self.transform = transform
        self.images = SampleIoHandler(args).load()  # list of (class_id, img_name, class_index)

    def __getitem__(self, index):
        name = self.images[index][1]
        label = self.images[index][2]
        image = self.transform(Image.open(os.path.join(self.image_root, self.images[index][0], name)).convert('RGB'))
        return os.path.splitext(name)[0], image, label # splitext will remove the suffix ".JPEG"

    def __len__(self):
        return len(self.images)

class TinyImageNet_selected(Dataset):

    def __init__(self, args, root: str, image_root: str, transform: torchvision.transforms.Compose) -> None:
        super().__init__()
        self.root = root
        self.image_root = image_root
        self.transform = transform
        self.images = SampleIoHandler(args).load()  # list of (class_id, img_name, class_index)
        # print(self.images)

    def __getitem__(self, index):
        name = self.images[index][1]
        label = self.images[index][2]
        image = self.transform(Image.open(os.path.join(self.image_root, self.images[index][0], "images", name)).convert('RGB'))
        return os.path.splitext(name)[0], image, label # splitext will remove the suffix ".JPEG"

    def __len__(self):
        return len(self.images)

class CIFAR10_selected(Dataset):

    def __init__(self, args, root: str, transform: torchvision.transforms.Compose, train: bool) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.selected_imgs = SampleIoHandler(args).load()  # list of (class_name, img index(in the WHOLE dataset, not in a specific class, 0-based), class_index)
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, transform=transform, download=True)
        # self.dataset_selected = Subset(self.dataset, self.selected_indices)
        self.class_name_list = self.dataset.classes # list of class names

    def __getitem__(self, index):
        img_index_in_whole_dataset = self.selected_imgs[index][1]
        image, label = self.dataset[img_index_in_whole_dataset] # call the __getitem__ method of dataset, label is int
        assert label == self.selected_imgs[index][2]
        assert self.class_name_list[label] == self.selected_imgs[index][0]
        name = self.class_name_list[label] + "_%05d" % img_index_in_whole_dataset # e.g. airplane_00029
        return name, image, label

    def __len__(self):
        return len(self.selected_imgs)


class CelebA_selected(Dataset): # only for gender estimation!!!  Note: index in the name is 1-based rather than 0-based

    def __init__(self, args, root: str, transform: torchvision.transforms.Compose, train: bool) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.selected_imgs = SampleIoHandler(args).load()  # list of (class_name, img index(in the WHOLE dataset, not in a specific class, *1-based*), class_index)
        self.split = "train" if train else "valid"
        self.dataset = torchvision.datasets.CelebA(root=root, split=self.split, target_type="attr", transform=transform, download=True)
        self.class_name_list = ["female", "male"] # 0 is female, 1 is male

    def __getitem__(self, index):
        img_index_in_whole_dataset = self.selected_imgs[index][1] # 1-based
        image, label = self.dataset[img_index_in_whole_dataset - 1] # since index is 1-based, we need to subtract 1 here. label is a tensor of shape (40,)
        assert label[20] == self.selected_imgs[index][2] # label[20] is the gender attribute
        assert self.class_name_list[label[20]] == self.selected_imgs[index][0]
        name = self.selected_imgs[index][0] + "_%06d" % (img_index_in_whole_dataset) # e.g. male_000003, note that it is 1-based (compatible with the CelebA naming scheme)
        return name, image, label

    def __len__(self):
        return len(self.selected_imgs)


class Tabular_selected(Dataset):
    def __init__(self, args, root: str, train: bool) -> None:
        super().__init__()
        self.root = root
        self.selected_samples = SampleIoHandler(args).load()  # list of (interger indices, class_index)
        split = "train" if train else "test"
        self.X = np.load(os.path.join(self.root, f"X_{split}.npy"))
        self.y = np.load(os.path.join(self.root, f"y_{split}.npy"))

    def __getitem__(self, index):
        index_in_whole_dataset = self.selected_samples[index][0]
        data = self.X[index_in_whole_dataset].astype(np.float32)
        label = self.y[index_in_whole_dataset].astype(np.int64)
        assert label == self.selected_samples[index][1] # label should match
        # name = f"data_idx{index}_real_idx{index_in_whole_dataset}" # legacy version, used in ICLR paper
        name = f"class{label}_real_idx{index_in_whole_dataset}"
        return name, data, label

    def __len__(self):
        return len(self.selected_samples) # call the __len__ method of Subset

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

# might be useless
def get_folder_class_mapping(args) -> List[Tuple[str, str]]:
    assert args.dataset in ['imagenet', 'tiny-imagenet', 'voc'], 'Dataset name should be imagenet, tiny-imagenet or voc'
    root = os.path.join(args.prefix, args.datasets_dirname, args.dataset_dirname)
    image_root = os.path.join(root, 'val')
    if args.dataset == 'imagenet':
        with open(os.path.join(root, 'folder_to_label.txt'), 'r', encoding='UTF-8') as f:
            return [tuple(line.strip().split(',')) for line in f.readlines()]  # line format: n02930766,出租车
    elif args.dataset == 'tiny-imagenet' or args.dataset == 'voc':
        return [(item, item) for item in sorted(os.listdir(image_root))]


