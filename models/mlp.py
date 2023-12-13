import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, dataset, n_layer):
        super(MLP, self).__init__()
        if dataset == 'census':
            in_dim = 12
            hidd_dim = 100
            out_dim = 2
        elif dataset == 'commercial':
            in_dim = 10
            hidd_dim = 100
            out_dim = 2
        else:
            raise Exception("dataset [%s] not implemented" % dataset)

        if n_layer < 2:
            raise Exception(f"Invalid #layer: {n_layer}.")

        self.all_layers = self._make_layers(in_dim, hidd_dim, out_dim, n_layer)

    def _make_layers(self, in_dim, hidd_dim, out_dim, n_layer):
        layers = [nn.Linear(in_dim, hidd_dim), nn.ReLU()]
        for _ in range(n_layer - 2):
            layers.extend([nn.Linear(hidd_dim, hidd_dim), nn.ReLU()])
        # layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(hidd_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(x.shape[0],-1)
        return self.all_layers(x)

# ===========================
#   wrapper
# ===========================
def mlp5(dataset):
    return MLP(dataset, n_layer=5)

def mlp8(dataset):
    return MLP(dataset, n_layer=8)

def mlp3(dataset):
    return MLP(dataset, n_layer=3)

if __name__ == '__main__':
    x = torch.rand(1000,10)
    net = mlp5("census")
    print(net)
    print(net(x).shape)