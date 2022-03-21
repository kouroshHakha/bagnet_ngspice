from cgl.utils.file import read_hdf5
import torch
import torch.nn as nn
import torch.nn.functional as F

from cgl.utils.params import ParamDict
from cgl.utils.torch import MLP
from .base import BaseNodeEmbeddingModule


class MLPBase(BaseNodeEmbeddingModule):

    def __init__(self, config: ParamDict) -> None:
        super().__init__(config)

    def get_input_struct(self, batch) -> ParamDict:
        x = torch.cat([batch.x, batch.type_tens], -1).to(self.device)
        bsize = len(batch.ptr) - 1
        num_input_nodes = batch.x.shape[0] // bsize
        return ParamDict(x=x.reshape(bsize, num_input_nodes * x.shape[-1]))

    def get_output_struct(self, batch) -> ParamDict:
        bsize = len(batch.ptr) - 1
        num_nodes = self.config.num_nodes

        output_shape = (bsize, num_nodes, -1)

        return ParamDict(vac_mag=batch.vac_mag.reshape(*output_shape), 
                         vac_ph=batch.vac_ph.reshape(*output_shape),
                         vdc=batch.vdc.reshape(*output_shape))
    

class MLPSharedBody(MLPBase):

    def __init__(self, config: ParamDict) -> None:
        config.setdefault('with_bn', False)
        super().__init__(config)

    def build_network(self, config):
        self.node_nets = torch.nn.ModuleList()
        self.shared_net = MLP(config.in_channels, config.hidden_channels, config.hidden_channels, config.num_shared_layers, config.dropout, bn=config.with_bn)
        for _ in range(config.num_nodes):
            self.node_nets.append(
                MLP(config.hidden_channels, config.hidden_channels, config.hidden_channels, config.num_node_feat_layers, config.dropout, bn=config.with_bn)
            )

        self.proj = torch.nn.Linear(config.hidden_channels, config.n_freqs * 2 + 1)
        self.n_freqs = config.n_freqs

    def reset_parameters(self):
        self.shared_net.reset_parameters()
        for net in self.node_nets:
            net.reset_parameters()
        self.proj.reset_parameters()

    def get_node_features(self, inputs: ParamDict) -> torch.Tensor:
        x = F.relu(self.shared_net(inputs.x))
        node_embs = [F.relu(net(x)) for net in self.node_nets]
        node_embs = torch.stack(node_embs, 1)
        return node_embs

    def project_node_to_output(self, inputs: ParamDict) -> ParamDict:
        projected_nodes = [self.proj(inputs.node_embs[:, i, :]) for i in range(inputs.node_embs.shape[1])]
        projected_nodes = torch.stack(projected_nodes, 1)

        output = ParamDict(
            node_embs=inputs.node_embs,
            vac_mag=projected_nodes[..., :self.n_freqs],
            vac_ph=projected_nodes[..., self.n_freqs:2*self.n_freqs],
            vdc=projected_nodes[..., -1:]
        )

        return output


class MLPSharedBodyDeepOutputHead(MLPSharedBody):

    def __init__(self, config: ParamDict) -> None:
        super().__init__(config)

    def build_network(self, config):
        super().build_network(config)
        self.proj = MLP(config.hidden_channels, config.hidden_channels, 
                        config.n_freqs * 2 + 1, config.num_out_head_layers, 
                        config.dropout, bn=config.with_bn)


class MLPPyramid(MLPBase):

    def __init__(self, config: ParamDict) -> None:
        super().__init__(config)

    def build_network(self, config):
        self.net = MLP(config.in_channels, config.hidden_channels, 
                       config.num_nodes * (config.n_freqs * 2 + 1), config.num_layers, 
                       config.dropout, bn=config.with_bn)
        self.n_freqs = config.n_freqs

    def reset_parameters(self):
        self.net.reset_parameters()

    def forward(self, inputs: ParamDict):
        out = self.net(inputs.x)
        out = out.reshape(-1, self.config.num_nodes, self.config.n_freqs * 2 + 1)
        output = ParamDict(
            vac_mag=out[..., :self.n_freqs],
            vac_ph=out[..., self.n_freqs:2*self.n_freqs],
            vdc=out[..., -1:]
        )
        return output

class PCA(nn.Module):

    def __init__(self, pca_params_file: str):
        super().__init__()

        content = read_hdf5(pca_params_file)
        self.weights = nn.Parameter(torch.from_numpy(content['weight']), requires_grad=False)
        if 'mean' in content:
            self.mean = nn.Parameter(torch.from_numpy(content['mean']), requires_grad=False)
        else:
            self.mean = None
        self.out_dim = self.weights.shape[0]
    
    def forward(self, x):
        if self.mean is not None:
            x = x - self.mean
        return torch.matmul(x, self.weights.T)

class MLPPyramidWithPCA(MLPBase):

    def __init__(self, config: ParamDict) -> None:
        super().__init__(config)

    def build_network(self, config):
        self.pca = PCA(config.pca_file)
        self.net = MLP(self.pca.out_dim, config.hidden_channels, 
                       config.num_nodes * (config.n_freqs * 2 + 1), config.num_layers, 
                       config.dropout, bn=config.with_bn)
        self.n_freqs = config.n_freqs

    def reset_parameters(self):
        self.net.reset_parameters()

    def forward(self, inputs: ParamDict):
        x = self.pca(inputs.x)
        out = self.net(x)
        out = out.reshape(-1, self.config.num_nodes, self.config.n_freqs * 2 + 1)
        output = ParamDict(
            vac_mag=out[..., :self.n_freqs],
            vac_ph=out[..., self.n_freqs:2*self.n_freqs],
            vdc=out[..., -1:]
        )
        return output

# class MLPResBlocks(nn.Module):

#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         self.lin1(x)

# class MLPSharedBodyWithResNets(MLPSharedBody):

#     def __init__(self, config: ParamDict) -> None:
#         super(BaseNodeEmbeddingModule).__init__(config)

#         self.node_nets = torch.nn.ModuleList()
#         self.shared_net = MLP(config.in_channels, config.hidden_channels, config.hidden_channels, config.num_shared_layers, config.dropout)
#         for _ in range(config.num_nodes):
#             self.node_nets.append(
#                 MLP(config.hidden_channels, config.hidden_channels, config.hidden_channels, config.num_out_head_layers, config.dropout)
#             )

#         self.proj = torch.nn.Linear(config.hidden_channels, config.n_freqs * 2 + 1)
#         self.n_freqs = config.n_freqs

#     def reset_parameters(self):
#         self.shared_net.reset_parameters()
#         for net in self.node_nets:
#             net.reset_parameters()
#         self.proj.reset_parameters()

#     def get_node_features(self, inputs: ParamDict) -> torch.Tensor:
#         x = F.relu(self.shared_net(inputs.x))
#         node_embs = [F.relu(net(x)) for net in self.node_nets]
#         node_embs = torch.stack(node_embs, 1)
#         return node_embs

