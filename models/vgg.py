"""vgg in pytorch
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):

    def __init__(self, features, dataset="tiny50"):
        super().__init__()
        self.features = features
        if dataset == 'tiny50':
            out_size = 7
            num_class = 50
        elif dataset == 'tiny10':
            out_size = 7
            num_class = 10
        elif dataset == 'cifar10':
            out_size = 1
            num_class = 10
        else:
            raise Exception("dataset [%s] not implemented" % dataset)

        self.classifier = nn.Sequential(
            # nn.AdaptiveAvgPool2d((out_size, out_size)), # we do not need this, because all images are resized properly
            nn.Flatten(),
            nn.Linear(512 * out_size * out_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

        self._initialize_weights() # rqh 0821 add

    def forward(self, x):
        output = self.features(x)
        output = self.classifier(output)

        return output

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11(dataset):
    return VGG(make_layers(cfg['A'], batch_norm=False), dataset=dataset)

def vgg13(dataset):
    return VGG(make_layers(cfg['B'], batch_norm=False), dataset=dataset)

def vgg16(dataset):
    return VGG(make_layers(cfg['D'], batch_norm=False), dataset=dataset)

def vgg19(dataset):
    return VGG(make_layers(cfg['E'], batch_norm=False), dataset=dataset)


# rqh 0629 add
class our_VGG(nn.Module): # put all layers in sequential

    def __init__(self, conv_layers, dataset):
        """
        dropout_layer:
        - -1: no dropout
        - 0: apply pixel-wise dropout to the input image
        for vgg11:
        - 2: apply dropout to the feature after the 1st conv-relu
        - 5: apply dropout to the feature after the 2rd conv-relu
        - 6: apply dropout to the feature after the 2rd max pooling
        for vgg16:
        - 4: apply dropout to the feature after the 2rd conv-relu
        - 5: apply dropout to the feature after the 1st max pooling
        """
        super().__init__()
        if dataset == 'tiny50':
            out_size = 7
            num_class = 50
            fc_dim = 4096
        elif dataset == 'tiny10' or dataset == "tiny10-lmj":
            out_size = 7
            num_class = 10
            fc_dim = 4096
        elif dataset == 'cifar10':
            out_size = 1
            num_class = 10
            fc_dim = 4096
        elif 'celeba' in dataset:
            out_size = 7
            num_class = 2 # todo: softmax or sigmoid? does it matter?
            fc_dim = 4096
        else:
            raise Exception("dataset [%s] not implemented" % dataset)

        self.all_layers = nn.Sequential(
            *conv_layers,
            nn.Flatten(),
            nn.Linear(512 * out_size * out_size, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc_dim, num_class)
        )

        self._initialize_weights()  # rqh 0821 add

    def forward(self, x):
        output = self.all_layers(x) # rqh 0902 delete dropout in forward, if dropout needs to be applied, do it using model.all_layer[:dp_layer] then dropout
        return output

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def our_vgg11(dataset):
    return our_VGG(make_layers(cfg['A'], batch_norm=False), dataset=dataset)
def our_vgg16(dataset):
    return our_VGG(make_layers(cfg['D'], batch_norm=False), dataset=dataset)
def our_vgg19(dataset):
    return our_VGG(make_layers(cfg['E'], batch_norm=False), dataset=dataset)

# rqh 0630 add
class our_pure_VGG(nn.Module): # put all layers in sequential, and delete the original dropout layers

    def __init__(self, conv_layers, dataset):
        """
        dropout_layer:
        - -1: no dropout
        - 0: apply pixel-wise dropout to the input image
        for vgg11:
        - 2: apply normal dropout to the feature after the 1st conv-relu
        - 5: apply normal dropout to the feature after the 2rd conv-relu
        - 6: apply normal dropout to the feature after the 2rd max pooling
        for vgg16:
        - 4: apply normal dropout to the feature after the 2rd conv-relu
        """
        super().__init__()

        if dataset == 'tiny50':
            out_size = 7
            num_class = 50
            fc_dim = 4096
        elif dataset == 'tiny10':
            out_size = 7
            num_class = 10
            fc_dim = 4096
        elif dataset == 'cifar10':
            out_size = 1
            num_class = 10
            fc_dim = 4096
        else:
            raise Exception("dataset [%s] not implemented" % dataset)

        self.all_layers = nn.Sequential(
            *conv_layers,
            nn.Flatten(),
            nn.Linear(512 * out_size * out_size, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, num_class)
        )

        self._initialize_weights() # rqh 0821 add

    def forward(self, x):
        output = self.all_layers(x) # rqh 0902 delete dropout in forward, if dropout needs to be applied, do it using model.all_layer[:dp_layer] then dropout
        return output

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def our_pure_vgg11(dataset):
    return our_pure_VGG(make_layers(cfg['A'], batch_norm=False), dataset=dataset)

def our_pure_vgg16(dataset):
    return our_pure_VGG(make_layers(cfg['D'], batch_norm=False), dataset=dataset)

if __name__ == '__main__':
    model = our_vgg19("tiny10")
    # print(model.all_layers)
    print(model.all_layers[:0])
    print(model.all_layers[0:])
    x = torch.randn(1,3,224,224)
    output_0 = model.all_layers[:0](x)
    x_clone = x.clone().detach()
    print(torch.norm(x - output_0))