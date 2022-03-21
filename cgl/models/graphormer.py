
"""
WIP:

1) At the very beggining attn_bias is not really differentiating between nodes with similar features but different structures. 
So I have a hypothesis that this remains true during the optimization due to initialization of spatial embs, so I changed the initiliazation from (0, 0.02) [default] to (0, 1.0)
to see what will happen? I also want to stop at step=10k to see if attn_bias is still not differentiating.

"""
# from data import get_dataset
import math

import networkx as nx

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

import torch_geometric as PyG
from torch_geometric.data import InMemoryDataset

from cgl.models.base import BaseNodeEmbeddingModule
from cgl.utils.params import ParamDict
from cgl.utils.torch import MLP


def get_graphormer_dset_wrapper(base):
    class GraphormerDataset(base):

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

            # modify the dataset to include degree and spatial info
            # these two are shared across all graphs for our specific problem of interest
            sample_data = super().get(0)
            nxg = PyG.utils.to_networkx(sample_data, to_undirected=True)
            self.degrees = torch.Tensor([nd for _, nd in nxg.degree])[:, None].long()
            self.spatial_pos = -torch.ones((len(sample_data.x), len(sample_data.x)), dtype=torch.long)
            for row in nx.shortest_path_length(nxg):
                row_idx = row[0]
                for col_idx, sp_len in row[1].items():
                    self.spatial_pos[row_idx, col_idx] = sp_len

            assert torch.all(self.spatial_pos != -1), 'Something is wrong with the graph data.' 
        
        def get(self, idx):
            data = super().get(idx)
            data.degrees = self.degrees
            data.spatial_pos = self.spatial_pos
            return data

    return GraphormerDataset

class PolynomialDecayLR(_LRScheduler):

    def __init__(self, optimizer, warmup_updates, tot_updates, lr, end_lr, power, last_epoch=-1, verbose=False):
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            self.warmup_factor = self._step_count / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif self._step_count >= self.tot_updates:
            lr = self.end_lr
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (
                self.tot_updates - warmup
            )
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr

        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        # module.weight.data.normal_(mean=0.0, std=0.02)
        module.weight.data.normal_(mean=0.0, std=1.0)


class Graphormer(BaseNodeEmbeddingModule):

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
        
        return super().configure_optimizers()

    def build_network(self, config):
        self.output_dim = sum(list(self.output_labels.values()))
        self.lin = nn.Linear(config.in_channels, config.hidden_channels)
        

        self.degree_emb = nn.Embedding(64, config.hidden_channels, padding_idx=0)
        self.spatial_pos_emb = nn.Embedding(40, config.nhead, padding_idx=0)

        encoders = [EncoderLayer(config.hidden_channels, config.hidden_channels, config.dropout, config.dropout, config.nhead)
                    for _ in range(config.num_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(config.hidden_channels)

        # if mlp_proj is set, the projection head will be an mlp, o.w. it is a linear layer.
        proj_n_layers = config.get('proj_n_layers', 1)
        self.proj = MLP(config.hidden_channels, 
                        config.hidden_channels, 
                        self.output_dim, 
                        num_layers=proj_n_layers, 
                        bn=True)

    def reset_parameters(self):
        self.apply(lambda module: init_params(module, n_layers=self.config.num_layers))
    
    def get_input_struct(self, batch) -> ParamDict:
        x = torch.cat([batch.x, batch.type_tens], -1).to(self.device)
        bsize = len(batch.ptr) - 1
        return ParamDict(
            x=x, 
            output_mask=batch.output_node_mask, #.reshape(bsize, num_input_nodes),
            batch_size=bsize,
            n_node_per_graph=batch.ptr[1],
            degrees=batch.degrees,
            spatial_pos=batch.spatial_pos,
            data=batch,
        )

    def get_node_features(self, inputs: ParamDict) -> torch.Tensor:
        # dataset should be modified to include degree and spatial pos

        feats = self.lin(inputs.x)
        bsize = inputs.batch_size
        n_node_per_graph = inputs.n_node_per_graph

        # spatial_pos: (b x n_node_per_graph, n_node_per_graph)
        spatial_pos = inputs.spatial_pos.view(bsize, n_node_per_graph, n_node_per_graph)
        # shape: (b, n_node_per_graph, n_node_per_graph, heads)
        # -> expected (b, heads, n_node_per_graph, n_node_per_graph)
        attn_bias = self.spatial_pos_emb(spatial_pos).permute(0, 3, 1, 2)

        for idx, layer in enumerate(self.layers):
            dim = feats.shape[1]

            # degrees shape: (b x n_node_per_graph, d)
            updated_feats = feats + self.degree_emb(inputs.degrees.squeeze(-1))

            # input shape to encoder: (n_graph, n_node_per_graph, d) = (B, Seq, d)
            encoded = layer(updated_feats.view(bsize, -1, dim) , attn_bias) #, bp_flag=self.training and self.global_step > 10000)
            if idx == len(self.layers) - 1:
                encoded = self.final_ln(encoded)
            feats = encoded.view(-1, dim)

        return feats[inputs.output_mask]        

    def get_output_struct(self, batch) -> ParamDict:
        # bsize = len(batch.ptr) - 1
        # num_nodes = self.config.num_nodes

        # output_shape = (bsize, num_nodes, -1)
        # return ParamDict(vdc=batch.vdc.reshape(*output_shape))
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

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None, bp_flag=False):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        # # TODO: remove
        if bp_flag:
            soft_qkt = torch.matmul(q, k).softmax(-1)
            soft_qkt_p = (torch.matmul(q, k) + attn_bias).softmax(-1)
            breakpoint()
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None, bp_flag=False):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias, bp_flag=bp_flag)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class GraphormerIndAttnBias(Graphormer):

    def build_network(self, config):
        super().build_network(config)
        self.spatial_pos_emb_list = nn.ModuleList([nn.Embedding(40, config.nhead, padding_idx=0) for _ in range(config.num_layers)])
    
    def get_node_features(self, inputs: ParamDict) -> torch.Tensor:
        # dataset should be modified to include degree and spatial pos

        feats = self.lin(inputs.x)
        bsize = inputs.batch_size
        n_node_per_graph = inputs.n_node_per_graph

        # spatial_pos: (b x n_node_per_graph, n_node_per_graph)
        spatial_pos = inputs.spatial_pos.view(bsize, n_node_per_graph, n_node_per_graph)


        for idx, (layer, spatial_emb) in enumerate(zip(self.layers, self.spatial_pos_emb_list)):
            dim = feats.shape[1]

            # degrees shape: (b x n_node_per_graph, d)
            updated_feats = feats + self.degree_emb(inputs.degrees.squeeze(-1))

            # shape: (b, n_node_per_graph, n_node_per_graph, heads)
            # -> expected (b, heads, n_node_per_graph, n_node_per_graph)
            attn_bias = spatial_emb(spatial_pos).permute(0, 3, 1, 2)
            # input shape to encoder: (n_graph, n_node_per_graph, d) = (B, Seq, d)
            encoded = layer(updated_feats.view(bsize, -1, dim) , attn_bias) #, bp_flag=self.training and self.global_step > 10000)
            if idx == len(self.layers) - 1:
                encoded = self.final_ln(encoded)
            feats = encoded.view(-1, dim)

        return feats[inputs.output_mask]       
