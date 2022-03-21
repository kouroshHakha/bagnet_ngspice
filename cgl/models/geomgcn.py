""" This file is compatible with dependencies in geom_gcn repo"""
import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.nn as PyGnn

import dgl
import dgl.function as fn

from cgl.utils.params import ParamDict

from cgl.models.gnn import GNNBase
import pdb



class GeomGCNSingleChannel(nn.Module):
    def __init__(self, in_feats, out_feats, num_divisions, activation, dropout_prob, merge):
        super(GeomGCNSingleChannel, self).__init__()
        self.num_divisions = num_divisions
        self.in_feats_dropout = nn.Dropout(dropout_prob)

        self.linear_for_each_division = nn.ModuleList()
        for i in range(self.num_divisions):
            self.linear_for_each_division.append(nn.Linear(in_feats, out_feats, bias=False))

        for i in range(self.num_divisions):
            nn.init.xavier_uniform_(self.linear_for_each_division[i].weight)

        self.activation = activation
        # self.g = g
        # self.subgraph_edge_list_of_list = self.get_subgraphs(self.g)
        self.merge = merge
        self.out_feats = out_feats

    def get_subgraphs(self, g):
        subgraph_edge_list = [[] for _ in range(self.num_divisions)]
        u, v, eid = g.all_edges(form='all')
        for i in range(g.number_of_edges()):
            subgraph_edge_list[g.edges[u[i], v[i]].data['subgraph_idx']].append(eid[i])

        return subgraph_edge_list

    def forward(self, g, feature):
        in_feats_dropout = self.in_feats_dropout(feature)
        subgraph_edge_list_of_list = self.get_subgraphs(g)

        self.g.ndata['h'] = in_feats_dropout

        for i in range(self.num_divisions):
            subgraph = self.g.edge_subgraph(subgraph_edge_list_of_list[i])
            subgraph.copy_from_parent()
            subgraph.ndata[f'Wh_{i}'] = self.linear_for_each_division[i](subgraph.ndata['h']) * subgraph.ndata['norm']
            subgraph.update_all(message_func=fn.copy_u(u=f'Wh_{i}', out=f'm_{i}'),
                                reduce_func=fn.sum(msg=f'm_{i}', out=f'h_{i}'))
            subgraph.ndata.pop(f'Wh_{i}')
            subgraph.copy_to_parent()

        self.g.ndata.pop('h')
        
        results_from_subgraph_list = []
        for i in range(self.num_divisions):
            if f'h_{i}' in self.g.node_attr_schemes():
                results_from_subgraph_list.append(self.g.ndata.pop(f'h_{i}'))
            else:
                results_from_subgraph_list.append(
                    torch.zeros((feature.size(0), self.out_feats), dtype=torch.float32, device=feature.device))

        if self.merge == 'cat':
            h_new = torch.cat(results_from_subgraph_list, dim=-1)
        else:
            h_new = torch.mean(torch.stack(results_from_subgraph_list, dim=-1), dim=-1)
        h_new = h_new * self.g.ndata['norm']
        h_new = self.activation(h_new)
        return h_new

class GeomGCNDGL(nn.Module):
    def __init__(self, in_feats, out_feats, num_divisions, activation, num_heads, dropout_prob, ggcn_merge,
                 channel_merge):
        super(GeomGCNDGL, self).__init__()
        self.attention_heads = nn.ModuleList()
        for _ in range(num_heads):
            self.attention_heads.append(
                GeomGCNSingleChannel(in_feats, out_feats, num_divisions, activation, dropout_prob, ggcn_merge))
        self.channel_merge = channel_merge

    def forward(self, g, feature):
        all_attention_head_outputs = [head(g, feature) for head in self.attention_heads]
        if self.channel_merge == 'cat':
            return torch.cat(all_attention_head_outputs, dim=1)
        else:
            return torch.mean(torch.stack(all_attention_head_outputs), dim=0)


class GeomGCNNetDGL(nn.Module):
    def __init__(self, num_input_features, num_output_classes, num_hidden, num_divisions, num_heads_layer_one,
                 num_heads_layer_two,
                 dropout_rate, layer_one_ggcn_merge, layer_one_channel_merge, layer_two_ggcn_merge,
                 layer_two_channel_merge):
        super(GeomGCNNetDGL, self).__init__()
        self.geomgcn1 = GeomGCNDGL(num_input_features, num_hidden, num_divisions, F.relu, num_heads_layer_one,
                                dropout_rate,
                                layer_one_ggcn_merge, layer_one_ggcn_merge)

        if layer_one_ggcn_merge == 'cat':
            layer_one_ggcn_merge_multiplier = num_divisions
        else:
            layer_one_ggcn_merge_multiplier = 1

        if layer_one_channel_merge == 'cat':
            layer_one_channel_merge_multiplier = num_heads_layer_one
        else:
            layer_one_channel_merge_multiplier = 1

        self.geomgcn2 = GeomGCNDGL(num_hidden * layer_one_ggcn_merge_multiplier * layer_one_channel_merge_multiplier,
                                num_output_classes, num_divisions, lambda x: x,
                                num_heads_layer_two, dropout_rate, layer_two_ggcn_merge, layer_two_channel_merge)

    def forward(self, g, features):
        x = self.geomgcn1(g, features)
        x = self.geomgcn2(g, x)
        return x

class GeomGCN(GNNBase):

    def __init__(self, config: ParamDict) -> None:
        super().__init__(config)

    def build_network(self, config):
        super().build_network(config)
        self.lin = nn.Linear(config.in_channels, config.hidden_channels)
        self.dgl_geom_gcn = GeomGCNNetDGL(
            config.hidden_channels,
            config.hidden_channels,
            config.hidden_channels,
            num_divisions=1,
            num_heads_layer_one=1,
            num_heads_layer_two=1,
            dropout_rate=0,
            layer_one_ggcn_merge=True,
            layer_one_channel_merge=True,
            layer_two_ggcn_merge=True,
            layer_two_channel_merge=True,
        )

    def reset_parameters(self):
        super().reset_parameters()

    def _convert_data_to_dgl(self, data):
        g = dgl.graph((data.edge_index[0], data.edge_index[1]), num_nodes=data.num_nodes)
        g.ndata['x'] = data['x']
        return g

    def convert_batched_data_to_batch_dgl(self, batched_data):
        # convert batched Data to a list of Data
        data_list = batched_data.to_data_list()
        # convert the list of Data to a list of DGLGraph
        dgl_data_list = [self._convert_data_to_dgl(data) for data in data_list]
        # convert the list of DGLGraph to a batched DGLGraph
        return dgl.batch(dgl_data_list)

    def get_node_features(self, inputs: ParamDict) -> torch.Tensor:
        output_mask = inputs.data.output_node_mask
        hidden_x = inputs.data.x
        # convert inputs.data to dgl batched graph
        g = self.convert_batched_data_to_batch_dgl(inputs.data)

        hidden_x = self.lin(inputs.data.x)
        pdb.set_trace()
        hidden_x = self.dgl_geom_gcn(g, hidden_x)

        return hidden_x[output_mask]
