import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.nn as PyGnn

from cgl.utils.params import ParamDict
from cgl.utils.optim import CosineWarmupScheduler
from cgl.utils.torch import MLP
from cgl.models.base import BaseNodeEmbeddingModule


class GNNBase(BaseNodeEmbeddingModule):

    def __init__(self, config: ParamDict) -> None:
        config.setdefault('activation', 'relu')
        super().__init__(config)

    def configure_optimizers(self):
        lr_warmup = self.config.get('lr_warmup', {})
        if lr_warmup:
            optim = super().configure_optimizers()
            lr_scheduler = CosineWarmupScheduler(optim, **lr_warmup)
            return [optim], [lr_scheduler]
        return super().configure_optimizers()

    def build_network(self, config):
        self.output_dim = sum(list(self.output_labels.values()))

        # if mlp_proj is set, the projection head will be an mlp, o.w. it is a linear layer.
        proj_n_layers = config.get('proj_n_layers', 1)
        self.proj = MLP(config.hidden_channels, config.hidden_channels, self.output_dim, num_layers=proj_n_layers, bn=True)
        self.act = self.get_activation()

    def reset_parameters(self):
        self.proj.reset_parameters()

    def get_input_struct(self, batch) -> ParamDict:
        batch.x = torch.cat([batch.x, batch.type_tens], -1).to(self.device)
        return ParamDict(data=batch)

    def get_output_struct(self, batch) -> ParamDict:
        output = ParamDict(**{label: batch[label].to(self.device) for label in self.output_labels})
        return output

    def project_node_to_output(self, inputs: ParamDict) -> ParamDict:
        node_embs = inputs.node_embs
        projected_nodes = self.proj(node_embs)

        output = ParamDict(node_embs=inputs.node_embs)
        last_idx = 0
        for label, dim in self.output_labels.items():
            output[label] = projected_nodes[..., last_idx:last_idx+dim]
            last_idx += dim
        return output

    def get_activation(self):
        
        if self.config.activation is None:
            return None
        elif self.config.activation == 'relu':
            return nn.ReLU()
        elif self.config.activation == 'tanh':
            return nn.Tanh()
        elif self.config.activation == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.2)
        else:
            raise ValueError(f'Activation {self.config.activation} is not valid.')



class MixHopNN(GNNBase):

    def __init__(self, config: ParamDict) -> None:
        super().__init__(config)

    def build_network(self, config):
        super().build_network(config)
        self.lin = nn.Linear(config.in_channels, config.hidden_channels)
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            layer = PyGnn.GCNConv(config.hidden_channels, config.hidden_channels)
            self.layers.append(layer)

    def reset_parameters(self):
        super().reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def get_node_features(self, inputs: ParamDict) -> torch.Tensor:
        output_mask = inputs.data.output_node_mask
        hidden_x = inputs.data.x

        hidden_x = self.lin(inputs.data.x)
        for layer in self.layers:
            hidden_x = F.relu(layer(hidden_x, inputs.data.edge_index))

        return hidden_x[output_mask]