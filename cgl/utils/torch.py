import torch
import torch.nn as nn


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=0, activation=nn.ReLU(), bn=False):
        super(MLP, self).__init__()

        self.lins = []

        l1_out_dim = hidden_channels if num_layers > 1 else out_channels
        self.lins.append(torch.nn.Linear(in_channels, l1_out_dim))
        for i in range(num_layers-1):
            if bn:
                self.lins.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(activation)
            self.lins.append(nn.Dropout(dropout))
            out_dim = hidden_channels if i != num_layers - 2 else out_channels
            self.lins.append(torch.nn.Linear(hidden_channels, out_dim))

        self.nets = nn.Sequential(*self.lins)

    def reset_parameters(self):
        for layer in self.nets:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x):
        return self.nets(x)


class EMA(nn.Module):
    def __init__(self, alpha):
        super(EMA, self).__init__()
        self.alpha = alpha
        self.last_average = None
        
    def forward(self, x):
        if self.last_average is None:
            self.last_average = x
        else:
            self.last_average = (1 - self.alpha) * x + self.alpha * self.last_average
        return self.last_average