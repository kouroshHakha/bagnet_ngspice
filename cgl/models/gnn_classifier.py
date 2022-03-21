import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.nn as PyGnn

from cgl.utils.params import ParamDict
from cgl.utils.optim import CosineWarmupScheduler, PolynomialDecayLR
from cgl.utils.torch import MLP
from cgl.models.base import BaseNodeEmbeddingModule
from cgl.eval.evaluator import NodeClassEvaluator
from cgl.models.gnn import GNNBase


class GNNNodeClassifier(GNNBase):

    def __init__(self, config: ParamDict) -> None:
        super().__init__(config)
        self.evaluator = NodeClassEvaluator(config.bins)

    def get_loss(self, pred_output: ParamDict, true_output: ParamDict):
        for key in true_output:
            true_output[key] = true_output[key].type(pred_output[key].dtype)

        total = 0.0
        loss_dict, eval_dict = ParamDict(), ParamDict()
        for label in self.output_labels:
            ytrue = true_output[label]
            ypred = pred_output['{label}_logits']
            non_nan_inds = ~torch.isnan(ytrue[:, 0])
            loss = self._get_xent_loss(ytrue[non_nan_inds], ypred[non_nan_inds])
            loss_dict[label] = loss
            # TODO
            eval_dict[f'{label}_acc'] = self.evaluator.eval(dict(y_true=ytrue[non_nan_inds], y_pred=ypred[non_nan_inds]))
            total += loss

        loss_dict['total'] = total
        return loss_dict, eval_dict

    def build_network(self, config):
        self.output_dim = sum(list(self.output_labels.values()))

        # if mlp_proj is set, the projection head will be an mlp, o.w. it is a linear layer.
        proj_n_layers = config.get('proj_n_layers', 1)
        self.proj = MLP(config.hidden_channels, config.hidden_channels, self.config.bins, num_layers=proj_n_layers, bn=True)
        self.act = self.get_activation()

    def project_node_to_output(self, inputs: ParamDict) -> ParamDict:
        node_embs = inputs.node_embs
        projected_nodes = self.proj(node_embs)

        output = ParamDict(node_embs=inputs.node_embs)

        # TODO: handle multi-lable output later
        output['vdc_logits'] = projected_nodes
        output['vdc'] =  torch.argmax(projected_nodes, -1) / self.bins + 1 / 2 / self.bins
        return output

    def _get_xent_loss(self, ytrue, ypred):
        
        # ytrue is a scalar, convert it to a class of 1 ... self.bins. shape ~ (N,1)
        # assume max is 1 and min 0
        max_val = 1.
        min_val = 0.

        # class range is [0, 0.01), [0.01, 0.02), ..., [0.99, 1.0]
        bins = self.config.bins
        bin_vec = torch.linspace(min_val, max_val, bins + 1)
        class_range = torch.stack([bin_vec[:-1], bin_vec[1:]], -1)
        

        # get class ids 1 ... self.bins and their one hot
        cond = (ytrue < class_range[:, 1]) & (ytrue >= class_range[:, 0])
        cond[:, -1:] |= (ytrue == max_val) # handling last class properly
        target = torch.where(cond)[1]

        loss = nn.CrossEntropyLoss()(input=ypred, target=target)
        return loss


class DeepGENNetClassifier(GNNNodeClassifier):

    def __init__(self, config: ParamDict) -> None:
        super().__init__(config)

    def build_network(self, config):
        super().build_network(config)
        self.lin = nn.Linear(config.in_channels, config.hidden_channels)
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            # in_channels = config.in_channels if i == 0 else config.hidden_channels
            conv = PyGnn.GENConv(config.hidden_channels, config.hidden_channels, aggr='softmax',
                                 t=1.0, learn_t=True, num_layers=2)
            act = nn.ReLU()
            norm = nn.LayerNorm(config.hidden_channels)
            layer = PyGnn.DeepGCNLayer(conv, norm, act, block='res+', dropout=config.dropout, ckpt_grad=config.get('ckpt_grad', False))
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
            hidden_x = layer(hidden_x, inputs.data.edge_index)
        hidden_x = self.layers[0].act(self.layers[0].norm(hidden_x))

        return hidden_x[output_mask]


class GATNetClassifier(GNNNodeClassifier):
    # TODO: fix this
    def __init__(self, config: ParamDict) -> None:
        super().__init__(config)

    def build_network(self, config):
        super().build_network(config)
        self.lin = nn.Linear(config.in_channels, config.hidden_channels)
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            layer = PyGnn.GATConv(config.hidden_channels, config.hidden_channels, heads=8, concat=False)
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


class GCNNetClassifer(GNNNodeClassifier):

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
