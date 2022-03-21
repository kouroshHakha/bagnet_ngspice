import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as PyGnn

from cgl.utils.params import ParamDict
from cgl.utils.torch import MLP
from cgl.models.base import BaseNodeEmbeddingModule
from cgl.utils.optim import CosineWarmupScheduler, PolynomialDecayLR
from cgl.models.gnn import DeepGENNet


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class PointTransformer(BaseNodeEmbeddingModule):

    def __init__(self, config: ParamDict) -> None:
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

        self.lin = nn.Linear(config.in_channels, config.hidden_channels)
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            layer = PyGnn.GCNConv(config.hidden_channels, config.hidden_channels)
            self.layers.append(layer)

        layer = nn.TransformerEncoderLayer(
            config.hidden_channels, config.nhead, config.hidden_channels, 
            config.dropout, config.activation, batch_first=True
        )
        norm = None #nn.LayerNorm(config.hidden_channels)
        depth = config.transformer_depth
        self.encode = nn.TransformerEncoder(layer, depth, norm)

        # if mlp_proj is set, the projection head will be an mlp, o.w. it is a linear layer.
        proj_n_layers = config.get('proj_n_layers', 1)
        self.proj = MLP(config.hidden_channels, 
                        config.hidden_channels, 
                        self.output_dim, 
                        num_layers=proj_n_layers, 
                        bn=True)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.proj.reset_parameters()

    def get_input_struct(self, batch) -> ParamDict:
        x = torch.cat([batch.x, batch.type_tens], -1).to(self.device)
        bsize = len(batch.ptr) - 1
        return ParamDict(
            x=x, 
            output_mask=batch.output_node_mask, #.reshape(bsize, num_input_nodes),
            batch_size=bsize,
            data=batch,
        )

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

    def get_node_features(self, inputs: ParamDict) -> torch.Tensor:

        feats = self.lin(inputs.x)
        for layer in self.layers:
            feats = F.relu(layer(feats, inputs.data.edge_index))

        dim = feats.shape[1]
        # input should be (B, N, dim) since batch_first=True
        encoded = self.encode(feats.view(inputs.batch_size, -1, dim))

        return encoded.view(-1, dim)[inputs.output_mask]


class InterleavedPointTransformer(BaseNodeEmbeddingModule):

    def __init__(self, config: ParamDict) -> None:
        super().__init__(config)

    # def configure_optimizers(self):
    #     lr_warmup = self.config.get('lr_warmup', {})
    #     if lr_warmup:
    #         optim = super().configure_optimizers()
    #         lr_scheduler = CosineWarmupScheduler(optim, **lr_warmup)
    #         return dict(optimizer=optim, lr_scheduler=lr_scheduler)
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
        
        return super().configure_optimizers()

    def build_network(self, config):
        self.output_dim = sum(list(self.output_labels.values()))
        self.lin = nn.Linear(config.in_channels, config.hidden_channels)
        self.layers = nn.ModuleList()

        if config.pre_gnn_depth:
            for _ in range(config.pre_gnn_depth):
                layer = PyGnn.GCNConv(config.hidden_channels, config.hidden_channels)
                self.layers.append(layer)

        tf_proto_layer = nn.TransformerEncoderLayer(
            config.hidden_channels, config.nhead, config.hidden_channels, 
            config.dropout, config.activation, batch_first=True
        )
        norm = None #nn.LayerNorm(config.hidden_channels)

        for _ in range(config.depth):
            # stack gcn layers
            for _ in range(config.num_layers):
                layer = PyGnn.GCNConv(config.hidden_channels, config.hidden_channels)
                self.layers.append(layer)
            # follow by transformer
            depth = config.transformer_depth
            self.layers.append(nn.TransformerEncoder(tf_proto_layer, depth, norm))

        # if mlp_proj is set, the projection head will be an mlp, o.w. it is a linear layer.
        proj_n_layers = config.get('proj_n_layers', 1)
        self.proj = MLP(config.hidden_channels, 
                        config.hidden_channels, 
                        self.output_dim, 
                        num_layers=proj_n_layers, 
                        bn=True)


    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, PyGnn.GCNConv): layer.reset_parameters()
        self.proj.reset_parameters()

    def get_input_struct(self, batch) -> ParamDict:
        x = torch.cat([batch.x, batch.type_tens], -1).to(self.device)
        bsize = len(batch.ptr) - 1
        return ParamDict(
            x=x, 
            output_mask=batch.output_node_mask, #.reshape(bsize, num_input_nodes),
            batch_size=bsize,
            data=batch,
        )

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

    def get_node_features(self, inputs: ParamDict) -> torch.Tensor:

        feats = self.lin(inputs.x)

        layer_cnt = 1
        for layer in self.layers:
            if isinstance(layer, PyGnn.GCNConv):
                feats = F.relu(layer(feats, inputs.data.edge_index))
            elif isinstance(layer, nn.TransformerEncoder):
                dim = feats.shape[1]
                # input should be (B, N, dim) since batch_first=True

                # ##### Plotting attention weights
                # import matplotlib.pyplot as plt
                # from torch_geometric.utils import to_dense_adj

                # node_map = {v:k for k, v in self.config.node_str.items()}
                # selected_rows = [k for k in node_map if node_map[k].startswith('V_net')]
                # batched_data_list = inputs.data.to_data_list()
                # dist_matrix = self.config.dist_matrix[selected_rows]
                # src = feats.view(inputs.batch_size, -1, dim)
                # for i in range(5):
                #     sample_adj = to_dense_adj(batched_data_list[i].edge_index)[0]
                #     sample_adj = sample_adj.detach().cpu().numpy()[selected_rows]
                #     _, att_w_list = layer.layers[0].self_attn(src, src, src)
                #     sample_att_w = att_w_list[i].detach().cpu().numpy()[selected_rows]

                #     plt.close()
                #     _, ax = plt.subplots(figsize=(15,4))
                    
                #     fsize = 10
                #     ax.imshow(sample_att_w) #, cmap='hot')
                #     ax.set_yticks([k for k in range(len(selected_rows))])
                #     ax.set_yticklabels([node_map[k] for k in selected_rows], fontdict={'fontsize': fsize})

                #     ax.set_xticks([k for k in node_map])
                #     ax.set_xticklabels(node_map.values(), fontdict={'fontsize': fsize})

                #     # Loop over data dimensions and create text annotations.
                #     for rid in range(len(selected_rows)):
                #         for cid in range(len(node_map)):
                #             ax.text(cid, rid, dist_matrix[rid, cid], ha="center", va="center", color="r")
                    
                #     # # Rotate the tick labels and set their alignment.
                #     plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
                #     path = Path(f'att_w_img/opamp/layer_{layer_cnt}')
                #     path.mkdir(parents=True, exist_ok=True)
                #     plt.savefig(path / f'att_heat_map_{i}.png', dpi=250)

                encoded = layer(feats.view(inputs.batch_size, -1, dim))
                feats = encoded.view(-1, dim)
            else:
                raise ValueError("Invalid layer type")
            
            layer_cnt += 1

        return feats.view(-1, dim)[inputs.output_mask]
    

class DeepGENBlock(nn.Module):

    def __init__(self, hidden_dim, dropout, ckpt_grad=False):
        super().__init__()
        conv = PyGnn.GENConv(hidden_dim, hidden_dim, aggr='softmax',
                                t=1.0, learn_t=True, num_layers=2)
        act = nn.ReLU()
        norm = nn.LayerNorm(hidden_dim)
        self.layer = PyGnn.DeepGCNLayer(conv, norm, act, block='res+', dropout=dropout, ckpt_grad=ckpt_grad)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index):
        return self.layer(x, edge_index)

    
class InterleavedPointTransformerDeepGEN(InterleavedPointTransformer):

    def __init__(self, config: ParamDict) -> None:
        super().__init__(config)

    def build_network(self, config):
        self.output_dim = sum(list(self.output_labels.values()))
        self.lin = nn.Linear(config.in_channels, config.hidden_channels)
        self.layers = nn.ModuleList()
        ckpt_grad = self.config.get('ckpt_grad', False)

        if config.pre_gnn_depth:
            for _ in range(config.pre_gnn_depth):
                layer = DeepGENBlock(config.hidden_channels, config.dropout, ckpt_grad=ckpt_grad)
                self.layers.append(layer)

        tf_proto_layer = nn.TransformerEncoderLayer(
            config.hidden_channels, config.nhead, config.hidden_channels, 
            config.dropout, config.activation, batch_first=True
        )
        norm = None #nn.LayerNorm(config.hidden_channels)

        for _ in range(config.depth):
            # stack deepgen layers
            for _ in range(config.num_layers):
                layer = DeepGENBlock(config.hidden_channels, config.dropout, ckpt_grad=ckpt_grad)
                self.layers.append(layer)
            # follow by transformer
            depth = config.transformer_depth
            self.layers.append(nn.TransformerEncoder(tf_proto_layer, depth, norm))

        # if mlp_proj is set, the projection head will be an mlp, o.w. it is a linear layer.
        proj_n_layers = config.get('proj_n_layers', 1)
        self.proj = MLP(config.hidden_channels, 
                        config.hidden_channels, 
                        self.output_dim, 
                        num_layers=proj_n_layers, 
                        bn=True)

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, DeepGENNet): layer.reset_parameters()
        self.proj.reset_parameters()

    def get_node_features(self, inputs: ParamDict) -> torch.Tensor:

        feats = self.lin(inputs.x)

        layer_cnt = 1
        for layer in self.layers:
            if isinstance(layer, DeepGENBlock):
                feats = F.relu(layer(feats, inputs.data.edge_index))
            elif isinstance(layer, nn.TransformerEncoder):
                dim = feats.shape[1]
                encoded = layer(feats.view(inputs.batch_size, -1, dim))
                feats = encoded.view(-1, dim)
            else:
                raise ValueError("Invalid layer type")
            
            layer_cnt += 1
            
        return feats.view(-1, dim)[inputs.output_mask]


class InterleavedPointTransformerDeepGENSingleOutput(InterleavedPointTransformerDeepGEN):

    def build_network(self, config):
        super().build_network(config)

        # use positional embedding on input to generate a scalar value for each freq
        self.proj = MLP(config.hidden_channels, 
                        config.hidden_channels, 
                        1, 
                        num_layers=config.get('proj_n_layers', 1), 
                        bn=True)

        self.pos_emb = PositionalEncoding(config.hidden_channels, 
                                          dropout=config.dropout, 
                                          max_len=self.output_dim)

    def project_node_to_output(self, inputs: ParamDict) -> ParamDict:
        node_embs = inputs.node_embs

        # repeat node_embs to match freq shape (N, D) -> (N, F, D)
        x = torch.stack([node_embs for _ in range(self.output_dim)], 1)
        x = self.pos_emb(x)
        x = x.reshape(-1, x.shape[-1])
        projected_nodes = self.proj(x)
        projected_nodes = projected_nodes.reshape(-1, self.output_dim)

        output = ParamDict(node_embs=inputs.node_embs)
        last_idx = 0
        for label, dim in self.output_labels.items():
            output[label] = projected_nodes[..., last_idx:last_idx+dim]
            last_idx += dim
        return output


class TransformerEnc(BaseNodeEmbeddingModule):

    def __init__(self, config: ParamDict) -> None:
        super().__init__(config)

    def build_network(self, config):
        encoder_layer = nn.TransformerEncoderLayer(
            config.hidden_channels, 4, config.hidden_channels, 
            config.dropout, config.activation
        )
        # encoder_norm = nn.LayerNorm(config.hidden_channels)
        encoder_norm = None
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_layers, encoder_norm)
        self.lin = nn.Linear(config.in_channels, config.hidden_channels)
        self.proj = nn.Linear(config.hidden_channels, 1)

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.proj.reset_parameters()

    def get_input_struct(self, batch) -> ParamDict:
        x = torch.cat([batch.x, batch.type_tens], -1).to(self.device)
        bsize = len(batch.ptr) - 1
        num_input_nodes = batch.x.shape[0] // bsize
        return ParamDict(x=x.reshape(bsize, num_input_nodes, x.shape[-1]), 
                         output_mask=batch.output_node_mask.reshape(bsize, num_input_nodes))

    def get_output_struct(self, batch) -> ParamDict:
        bsize = len(batch.ptr) - 1
        num_nodes = self.config.num_nodes

        output_shape = (bsize, num_nodes, -1)

        return ParamDict(vdc=batch.vdc.reshape(*output_shape))

    def project_node_to_output(self, inputs: ParamDict) -> ParamDict:
        node_embs = inputs.node_embs
        output = ParamDict(vdc=self.proj(node_embs))
        return output

    def get_node_features(self, inputs: ParamDict) -> torch.Tensor:
        
        # there is no need for masking right now since the topology is fixed and the nodes have a rigid structure
        x = inputs.x.permute(1, 0, 2)
        xp = self.encoder(self.lin(x)) # (n_node, batch, dim)
        xp = xp.permute(1, 0, 2) # (batch, n_node, dim)

        # all nodes are masked similiarly
        mask = inputs.output_mask[0]
        node_embs = xp[:, mask, :] # (batch, n_node, dim)

        return node_embs
