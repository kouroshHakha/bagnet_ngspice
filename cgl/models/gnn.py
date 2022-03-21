import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.container import ModuleList
from torch.optim.lr_scheduler import MultiStepLR

import torch_geometric.nn as PyGnn
import pytorch_lightning as pl

from cgl.utils.params import ParamDict
from cgl.utils.optim import CosineWarmupScheduler, PolynomialDecayLR
from cgl.utils.torch import MLP
from cgl.models.base import BaseNodeEmbeddingModule
from cgl.eval.evaluator import NodeEvaluator

import torch_scatter

def get_activation(activation):
    
    if activation is None:
        return None
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    else:
        raise ValueError(f'Activation {activation} is not valid.')


def get_optimizer(lr, parameters):

        if isinstance(lr, dict):
            # if it's a dict it should specify the warm-up schedule parameters
            optimizer = torch.optim.AdamW(
                parameters, lr=lr['peak_lr'], weight_decay=lr['weight_decay'])
            lr_scheduler = {
                'scheduler': PolynomialDecayLR(
                    optimizer,
                    warmup_updates=lr['warmup_updates'],
                    tot_updates=lr['tot_updates'],
                    lr=lr['peak_lr'],
                    end_lr=lr['end_lr'],
                    power=1.0,
                ),
                'name': 'learning_rate',
                'interval': 'step',
                'frequency': 1,
            }
            return dict(optimizer=optimizer, scheduler=lr_scheduler)
        else:
            return torch.optim.AdamW(parameters, lr=lr)


class GNNBase(BaseNodeEmbeddingModule):

    def __init__(self, config: ParamDict) -> None:
        config.setdefault('activation', 'relu')
        config.setdefault('output_sigmoid', [])
        super().__init__(config)

    # def configure_optimizers(self):
    #     lr_warmup = self.config.get('lr_warmup', {})
    #     if lr_warmup:
    #         optim = super().configure_optimizers()
    #         lr_scheduler = CosineWarmupScheduler(optim, **lr_warmup)
    #         return [optim], [lr_scheduler]
    #     return super().configure_optimizers()

    def configure_optimizers(self):
        lr_warmup = self.config.get('lr_warmup', {})

        if lr_warmup:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=lr_warmup['peak_lr'], weight_decay=lr_warmup['weight_decay'])
            lr_scheduler = {
                'scheduler': PolynomialDecayLR(
                    optimizer,
                    warmup_updates=lr_warmup['warmup_updates'],
                    tot_updates=lr_warmup['tot_updates'],
                    lr=lr_warmup['peak_lr'],
                    end_lr=lr_warmup['end_lr'],
                    power=1.0,
                ),
                'name': 'learning_rate',
                'interval': 'step',
                'frequency': 1,
            }
            return [optimizer], [lr_scheduler]

        if self.config.get('finetune', False):
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)

            step_lr = self.config.get('stepLR', False)
            if step_lr:
                scheduler = MultiStepLR(optimizer, milestones=step_lr, gamma=0.5)
                return [optimizer], [scheduler]
            else:
                return optimizer
        
        return super().configure_optimizers()

    def build_network(self, config):
        self.output_dim = sum(list(self.output_labels.values()))

        # if mlp_proj is set, the projection head will be an mlp, o.w. it is a linear layer.
        proj_n_layers = config.get('proj_n_layers', 1)
        self.proj = MLP(config.hidden_channels, config.hidden_channels, self.output_dim, num_layers=proj_n_layers, bn=True)
        self.act = get_activation(self.config.activation)

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
            if label in self.config.output_sigmoid:
                output[label] = output[label].sigmoid()
            last_idx += dim
        return output


class DeepGENNet(GNNBase):

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

    def get_node_features(self, inputs: ParamDict, return_masked=True) -> torch.Tensor:
        output_mask = inputs.data.output_node_mask
        hidden_x = inputs.data.x

        hidden_x = self.lin(inputs.data.x)
        # print('='*30)
        # for key,val in self.layers[0].state_dict().items():
        #     print('-'*30 + f' {key}')
        #     print(val)
        for layer in self.layers:
            hidden_x = layer(hidden_x, inputs.data.edge_index)
        hidden_x = self.layers[0].act(self.layers[0].norm(hidden_x))

        return hidden_x[output_mask] if return_masked else hidden_x

class GATNet(GNNBase):

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


class GCNNet(GNNBase):

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

class XAttn(nn.Module):

    def __init__(self, hidden_dim, state_dim, input_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_net = nn.Linear(state_dim, hidden_dim)
        self.value_net = nn.Linear(input_dim, hidden_dim)
        self.key_net = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, state):
        # shape of in_seq: B, M, D
        # shape of query_state: B, N, D
        # output shape: B, N, D

        Bin, M, _ = x.shape
        Bq, N, _ = state.shape
        
        assert Bin == Bq
        B = Bin = Bq

        query = self.query_net(state) # (B, N, D)
        key = self.key_net(x) # (B, M, D)
        value = self.value_net(x) # (B, M, D)

        attn = torch.einsum('bnd,bmd->bnm', query, key) / (self.hidden_dim ** 0.5)
        assert attn.shape == (B, N, M)
        attn_score = attn.softmax(-1)  # (B, N, M)
        output = torch.einsum('bnm,bmd->bnd', attn_score, value)
        output = output.contiguous()

        return output

class Node2GraphEmb(nn.Module):
    
    def __init__(self, hidden_dim, n_layers, embedding_dim, node_dim):
        super().__init__()
        graph_emb_in = nn.Parameter(torch.randn(embedding_dim), requires_grad=True)
        self.register_buffer('graph_emb_in', graph_emb_in)

        assert embedding_dim % hidden_dim == 0
        chunks = embedding_dim // hidden_dim
        query_state = graph_emb_in.view(chunks, hidden_dim)
        self.register_buffer('query_state', query_state)

        self.xatt_layers = ModuleList()
        for _ in range(n_layers):
            self.xatt_layers.append(XAttn(hidden_dim, hidden_dim, node_dim))

    
    def forward(self, node_embs):
        # shape node_embs: B, N, Dn
        # output shape: B, Dg
        B, _, _ = node_embs.shape
        # repeat hidden query state across batch dimension
        state_shape = (B,) + self.query_state.shape
        state_stride = (0,) + self.query_state.stride()
        hidden_state = torch.as_strided(self.query_state, state_shape, state_stride)
        for i in range(len(self.xatt_layers)):
            hidden_state = self.xatt_layers[i](node_embs, hidden_state)
        graph_emb = hidden_state.reshape((B, -1))
        return graph_emb


class GraphRegression(pl.LightningModule):

    def __init__(self, config, gnn_backbone: BaseNodeEmbeddingModule) -> None:
        super().__init__()
        self.gnn = gnn_backbone
        self.config = config
        self.evaluator = NodeEvaluator(config.bins)
        self.output_label = self.config.output_label
        self.output_sigmoid = self.config.output_sigmoid

        self.save_hyperparameters(dict(config))
        # self.node_ids = [75, 76, 77, 78, 80, 81, 82n, 83, 0]
        # self.node_ids = list(range(60, 84)) + [0]
        # self.node_ids = range(84)
        
        self.use_pooling = self.config.get('use_pooling', False)

        if self.use_pooling:
            graph_emb_dim = config.hidden_channels
            self.node2graph = None
        else:
            graph_emb_dim = 256
            self.node2graph = Node2GraphEmb(16, 3, graph_emb_dim, config.hidden_channels)

        self.output_proj = MLP(
            in_channels=graph_emb_dim, #config.hidden_channels * len(self.node_ids),
            hidden_channels=config.hidden_channels, 
            out_channels=1, 
            num_layers=config.num_layers, 
            dropout=config.dropout, 
            activation=get_activation(config.activation),
            bn=True,
        )

    def configure_optimizers(self):
        """
        supports
        # frozen backbone
        # backbone with a different learning rate / schedule than the output head
        # backbone with the same learning rate and schedule as the output head
        """
        # lr_params = self.config.lr_params
        # backbone = lr_params.get('backbone', {})

        # optimizers = []
        # if backbone:
        #     optimizers.append(get_optimizer(backbone, self.gnn.parameters()))
        # optimizers.append(get_optimizer(lr_params['output'], self.output_proj.parameters()))

        # return optimizers
        lr_warmup = self.config.get('lr_warmup', {})

        if lr_warmup:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=lr_warmup['peak_lr'], weight_decay=lr_warmup['weight_decay'])
            lr_scheduler = {
                'scheduler': PolynomialDecayLR(
                    optimizer,
                    warmup_updates=lr_warmup['warmup_updates'],
                    tot_updates=lr_warmup['tot_updates'],
                    lr=lr_warmup['peak_lr'],
                    end_lr=lr_warmup['end_lr'],
                    power=1.0,
                ),
                'name': 'learning_rate',
                'interval': 'step',
                'frequency': 1,
            }
            return [optimizer], [lr_scheduler]

        return optim.AdamW(self.parameters(), lr=self.config.lr)

            

    def forward(self, batch):
        
        input_struct = self.gnn.get_input_struct(batch)
        node_embs = self.gnn.get_node_features(input_struct, return_masked=False)

        # a general prescription for handling non-homogenous graphs (w/ diff # of nodes)
        # counts = torch_scatter.scatter_sum(batch.output_node_mask.int(), batch.batch, dim=0)
        # group_ids = torch.tensor([[id]*count for id, count in enumerate(counts)]).view(-1).contiguous()
        # graph_embs = torch.stack([node_embs[group_ids == i].view(-1) for i in range(batch.num_graphs)], 0)
        # graph_embs = torch.cat([node_embs[i::84] for i in self.node_ids], dim=-1)

        n_nodes = len(batch.x) // batch.num_graphs
        node_embs = torch.stack([node_embs[i::n_nodes] for i in range(n_nodes)], 1)
        # node_embs = torch.stack(torch.split(node_embs, batch.num_graphs, 0), dim=1)
        if self.use_pooling:
            graph_embs = node_embs.mean(1)
        else:
            graph_embs = self.node2graph(node_embs)
        output = self.output_proj(graph_embs)
        
        if self.output_sigmoid:
            return torch.sigmoid(output)
        return output

    def compute_ff(self, batch, return_loss=False):
        ypred = self.forward(batch)
        ypred = ypred.squeeze(-1)

        ytrue = batch[self.output_label]

        # ytrue = batch.rout_cls
        non_nan_inds = ~torch.isnan(ytrue)

        if return_loss:
            loss = nn.MSELoss()(ytrue[non_nan_inds], ypred[non_nan_inds])
            # loss = nn.BCELoss()(ypred[non_nan_inds], ytrue[non_nan_inds].float())
            # if self.training:
            #     grad = torch.autograd.grad(loss, self.gnn.lin.weight, retain_graph=True)[0]
            #     print(torch.linalg.norm(grad))
            acc = self.evaluator.eval(dict(y_true=ytrue[non_nan_inds], y_pred=ypred[non_nan_inds]))
            # acc = sum((ypred[non_nan_inds] > 0.5) == ytrue[non_nan_inds]) / len(ytrue[non_nan_inds])
            return dict(pred=ypred, loss=loss, acc=acc)

        return dict(pred=ypred)

    def training_step(self, batch, batch_idx):
        ret = self.compute_ff(batch, return_loss=True)
        output = dict(loss=ret['loss'], acc=ret['acc'])
        return output

    def validation_step(self, batch, batch_idx):
        ret = self.compute_ff(batch, return_loss=True)
        output = dict(loss=ret['loss'], acc=ret['acc'])
        return output  

    def test_step(self, batch, batch_idx):
        ret = self.compute_ff(batch, return_loss=True)
        output = dict(loss=ret['loss'], acc=ret['acc'])

        self.log('tloss', ret['loss'], on_epoch=True)
        self.log('tacc', ret['acc'], on_epoch=True)
        return output    
    
    def _summarize(self, outputs, prefix: str):
        loss_epoch = torch.tensor([output['loss'].item() for output in outputs]).mean()
        acc_epoch = torch.tensor([output['acc'].item() for output in outputs]).mean()

        self.log(f'{prefix}_loss_epoch', loss_epoch)
        self.log(f'{prefix}_acc_epoch', acc_epoch)

    def training_epoch_end(self, outputs) -> None:
        self._summarize(outputs, prefix='train')

    def validation_epoch_end(self, outputs) -> None:
        self._summarize(outputs, prefix='valid')



# class GCNNodeEmbedding(GNNBase):

#     def __init__(self, config: ParamDict) -> None:
#         super().__init__(config)

#         self.conv_list = nn.ModuleList()
#         self.bn_list = nn.ModuleList()
#         self.requires_activation = False

#         for i in range(config.num_layers):
#             in_channels = config.in_channels if i == 0 else config.hidden_channels
#             # # GCNConv
#             if self.config.gnn_core == 'GCNConv':
#                 self.conv_list.append(PyGnn.GCNConv(in_channels, config.hidden_channels))
#                 self.requires_activation = True
#             # # GINConv
#             elif self.config.gnn_core == 'GINConv':
#                 self.conv_list.append(
#                     PyGnn.GINConv(
#                         MLP(in_channels, config.hidden_channels, config.hidden_channels, num_layers=2, activation=self.act),
#                         train_eps=True
#                     )
#                 )
#             # # GENConv
#             elif self.config.gnn_core == 'GENConv':
#                 self.conv_list.append(PyGnn.GENConv(
#                     in_channels, config.hidden_channels, 
#                     num_layers=2, aggr='softmax', t=1.0, 
#                     learn_t=True, norm='layer')
#                 )
#             # # GAT
#             elif self.config.gnn_core == 'GATConv':
#                 self.conv_list.append(PyGnn.GATConv(in_channels, config.hidden_channels, heads=8, concat=False))
#                 self.requires_activation = True

#             self.bn_list.append(nn.BatchNorm1d(config.hidden_channels))

#         if self.requires_activation and self.act is None:
#             raise ValueError('Activation is required for multiple layers but it is not provided')

#         # if not requires_activation and self.act is not None:
#         #     raise ValueError('Activation should not be given as the conv layer already has the non-linearity')

#     def reset_parameters(self):
#         super().reset_parameters()
#         for conv_layer in self.conv_list:
#             conv_layer.reset_parameters()
#         for bn_layer in self.bn_list:
#             bn_layer.reset_parameters()

#     def get_node_features(self, inputs: ParamDict) -> torch.Tensor:
#         output_mask = inputs.data.output_node_mask
#         hidden_x = inputs.data.x
#         for bn_layer, conv_layer in zip(self.bn_list, self.conv_list):
#             hidden_x = bn_layer(conv_layer(hidden_x, inputs.data.edge_index))
#             if self.act is not None and self.requires_activation:
#                 hidden_x = self.act(hidden_x)
#         return hidden_x[output_mask]


# class MLPSimple(GNNBase):

#     """This is a fundamentally in applicable baseline. The nodes that we are interested in all 
#     share the same local features, but it's the context that determines their values and features.
#     Therefore applying the same MLP without feeding context / global information as input 
#     becomes in-applicable"""

#     def __init__(self, config: ParamDict) -> None:
#         super().__init__(config)

#         assert self.act is not None, 'You should pass in activation type'
#         self.nets = nn.ModuleList()

#         for i in range(config.num_layers):
#             in_channels = config.in_channels if i == 0 else config.hidden_channels
#             layer = nn.Linear(in_channels, config.hidden_channels)
#             self.nets.append(layer)

#     def reset_parameters(self):
#         super().reset_parameters()
#         for layer in self.nets:
#             layer.reset_parameters()

#     def get_node_features(self, inputs: ParamDict) -> torch.Tensor:
#         output_mask = inputs.data.output_node_mask
#         hidden_x = inputs.data.x
#         for fc in self.nets:
#             hidden_x = self.act(fc(hidden_x))
#         return hidden_x[output_mask]


# class GNNSimple(GNNBase):

#     def __init__(self, config: ParamDict) -> None:
#         super().__init__(config)

#     def build_network(self, config):
#         super().build_network(config)
#         self.conv_list = nn.ModuleList()
#         self.norm_layers = nn.ModuleList()
#         self.douts = nn.ModuleList()
#         self.layers = nn.ModuleList()
#         for i in range(config.num_layers):
#             in_channels = config.in_channels if i == 0 else config.hidden_channels
#             # conv = PyGnn.GENConv(in_channels, config.hidden_channels, aggr='softmax',
#             #                      t=1.0, learn_t=True, num_layers=2)
#             # act = nn.ReLU()
#             # norm = nn.LayerNorm(config.hidden_channels)
#             # layer = PyGnn.DeepGCNLayer(conv, norm, act, block='res+', dropout=config.dropout)
#             # self.layers.append(layer)
#             # self.conv_list.append(PyGnn.GATConv(in_channels, config.hidden_channels, aggr='mean', concat=False, heads=4))
#             self.conv_list.append(PyGnn.GCNConv(in_channels, config.hidden_channels, aggr='add'))
#             # self.norm_layers.append(nn.LayerNorm(config.hidden_channels))
#             self.norm_layers.append(nn.BatchNorm1d(config.hidden_channels))
#             self.douts.append(nn.Dropout(p=config.dropout))

#     def reset_parameters(self):
#         super().reset_parameters()
#         # for conv_layer in self.conv_list:
#         #     conv_layer.reset_parameters()
#         for layer in self.layers:
#             layer.reset_parameters()

#     def get_node_features(self, inputs: ParamDict) -> torch.Tensor:
#         output_mask = inputs.data.output_node_mask
#         hidden_x = inputs.data.x

#         # hidden_x = self.act(self.norm_layers[0](self.conv_list[0](hidden_x, inputs.data.edge_index)))
#         for i, conv_layer in enumerate(self.conv_list):
#             hidden_x = self.act(self.norm_layers[i](conv_layer(hidden_x, inputs.data.edge_index)))
#             # hidden_x = self.douts[i](self.act(conv_layer(hidden_x, inputs.data.edge_index)))
        
#         # hidden_x = self.layers[0].conv(inputs.data.x, inputs.data.edge_index)
#         # for layer in self.layers[1:]:
#         #     hidden_x = layer(hidden_x, inputs.data.edge_index)
#         # hidden_x = self.layers[0].act(self.layers[0].norm(hidden_x))
#         # hidden_x = F.dropout(hidden_x, p=self.config.dropout, training=self.training)

#         return hidden_x[output_mask]


# class TransformerSimple(GNNBase):

#     def __init__(self, config: ParamDict) -> None:
#         super().__init__(config)

#         encoder_layer = nn.TransformerEncoderLayer(config.hidden_channels, 4, config.hidden_channels, config.dropout, config.activation)
#         encoder_norm = nn.LayerNorm(config.hidden_channels)
#         self.encoder = nn.TransformerEncoder(encoder_layer, config.num_layers, encoder_norm)
#         self.input_lin = nn.Linear(config.in_channels, config.hidden_channels, bias=False)

#     def get_node_features(self, inputs: ParamDict) -> torch.Tensor:
#         # there is no need for masking right now since the topology is fixed and the nodes have a rigid structure
#         data_list = inputs.data.to_data_list()
        
#         x = torch.stack([data.x for data in data_list], 1) # (seq_len, batch, dim)
#         xp = self.encoder(self.input_lin(x)) # (seq_len, batch, dim)

#         # all nodes are masked similiarly
#         mask = data_list[0].output_node_mask
#         node_embs = xp[mask] # (n_node, batch, dim)
        
#         # swap batch and n_node dims, and flatten it so that it is (n_node * batch, dim)
#         node_embs = node_embs.permute(1, 0, 2)
#         node_embs = node_embs.reshape(-1, node_embs.shape[-1])


#         return node_embs


if __name__ == '__main__':

    B, D, T = 1, 128, 84
    node2graph = Node2GraphEmb(16, 3, 256, D)

    node_embs = torch.randn((B, T, D))
    graph_emb = node2graph(node_embs)
    breakpoint()