import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from collections import defaultdict

from cgl.utils.params import ParamDict
from cgl.utils.torch import MLP
from cgl.models.base import BaseNodeEmbeddingModule
from cgl.eval.evaluator import NodeEvaluator
from pytorch_lightning import LightningModule

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

        out_dict = ParamDict()
        for label, dim in self.output_labels.items():
            out_dict[label] = batch[label].reshape(*output_shape)
        return out_dict


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

        output_dim = sum(list(self.output_labels.values()))
        # if mlp_proj is set, the projection head will be an mlp, o.w. it is a linear layer.
        proj_n_layers = config.get('proj_n_layers', 1)
        self.proj = MLP(config.hidden_channels, config.hidden_channels, output_dim, num_layers=proj_n_layers, bn=True)
        

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

        output = ParamDict(node_embs=inputs.node_embs)
        last_idx = 0
        for label, dim in self.output_labels.items():
            output[label] = projected_nodes[..., last_idx:last_idx+dim]
            last_idx += dim
        return output


class MLPPyramid(MLPBase):

    def __init__(self, config: ParamDict) -> None:
        super().__init__(config)

    def build_network(self, config):
        self.output_dim = sum(list(self.output_labels.values()))
        self.net = MLP(config.in_channels, config.hidden_channels, 
                       config.num_nodes * self.output_dim, config.num_layers, 
                       config.dropout, bn=config.with_bn)

    def reset_parameters(self):
        self.net.reset_parameters()

    def forward(self, inputs: ParamDict):
        out = self.net(inputs.x)
        out = out.reshape(-1, self.config.num_nodes, self.output_dim)
        output = ParamDict()
        last_idx = 0
        for label, dim in self.output_labels.items():
            output[label] = out[..., last_idx:last_idx+dim]
            if label in self.config.output_sigmoid:
                output[label] = output[label].sigmoid()
            last_idx += dim
        return output


class MLPFixedInput(LightningModule):
    def __init__(self, config: ParamDict) -> None:
        super().__init__()
        self.save_hyperparameters()
        # self.output_labels = config.output_labels # {'vdc': 1, 'vac_mag': 101, 'vac_ph': 101}
        self.config = config
        self.evaluator = NodeEvaluator(config.bins)
        self.build_network(config)


    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config.lr)
        return optimizer

    def build_network(self, config):
        # self.output_dim = sum(list(self.output_labels.values()))
        self.net = MLP(config.in_channels, config.hidden_channels, 
                       config.num_nodes, config.num_layers, 
                       config.dropout, activation=nn.Tanh(), bn=config.with_bn)

    def reset_parameters(self):
        self.net.reset_parameters()

    def forward(self, x):
        out = self.net(x)
        return out
    
    def get_loss(self, y, yhat):
        total = 0.0
        loss_dict, eval_dict = ParamDict(), ParamDict()
        loss_dict['vdc'] = nn.MSELoss()(y, yhat)
        eval_dict[f'vdc_acc'] = self.evaluator.eval(dict(y_true=y, y_pred=yhat))
        total += loss_dict['vdc']

        loss_dict['total'] = total
        return loss_dict, eval_dict

    
    def training_step(self, batch, batch_idx):
        res = self._compute_ff(batch)
        output = dict(loss=res.loss.total, summary=res)
        return output

    def validation_step(self, batch, batch_idx):
        res = self._compute_ff(batch)
        output = dict(loss=res.loss.total, summary=res)
        return output

    def validation_epoch_end(self, outputs) -> None:
        # this val_loss_ema is used for monitoring for checkpoints and to avoid noisy validation loss
        # we do this in epoch end because we want to initialize it to epoch's validation not batch validation in step zero
        # total_val_losses = torch.stack([loss['summary'] for loss in outputs])
        # val_loss_ema = self.val_ema(torch.mean(total_val_losses))
        # self.log('valid_loss_total_ema', val_loss_ema)

        loss_epoch, eval_dict = self._get_loss_eval_epochs(outputs)
        self._log_dict_with_prefix(eval_dict, 'valid_epoch')
        self.log('valid_loss_epoch', loss_epoch)

    def training_epoch_end(self, outputs) -> None:
        loss_epoch, eval_dict = self._get_loss_eval_epochs(outputs)
        self._log_dict_with_prefix(eval_dict, 'train_epoch')
        self.log('train_loss_epoch', loss_epoch)

    def test_step(self, batch, batch_idx):
        res = self._compute_ff(batch)
        self._log_dict_with_prefix(res.loss, 'loss', on_epoch=True)
        self._log_dict_with_prefix(res.eval, on_epoch=True)

    def predict(self, batch, compute_loss=False):
        self.eval()
        with torch.no_grad():
            return self._compute_ff(batch, compute_loss)

    def _compute_ff(self, batch, compute_loss=True):
        x, y = batch
        yhat = self(x)
        if compute_loss:
            loss_dict, eval_dict = self.get_loss(y, yhat)
            return ParamDict(input=x, output=yhat, loss=loss_dict, eval=eval_dict)
        return ParamDict(input=x, output=yhat)

    
    def _log_dict_with_prefix(self, dictionary, prefix='', **kwargs):

        if not prefix:
            self.log_dict(dictionary, **kwargs)
        else:
            for key in dictionary:
                self.log(f'{prefix}_{key}', dictionary[key], **kwargs)


    def _get_loss_eval_epochs(self, outputs):
        loss_epoch = torch.stack([output['loss'] for output in outputs]).mean()

        eval_keys = outputs[0]['summary']['eval'].keys()
        eval_dict = defaultdict(lambda: [])
        for output in outputs:
            for key in eval_keys:
                eval_dict[key].append(output['summary']['eval'][key].item())
        eval_dict = {k: torch.tensor(v).mean() for k, v in eval_dict.items()}

        return loss_epoch, eval_dict


# class MLPNodeEmbedding(BaseNodeEmbeddingModule):

#     def __init__(self, config: ParamDict) -> None:
#         super().__init__(config)

#         self.node_nets = torch.nn.ModuleList()

#         for _ in range(config.num_nodes):
#             self.node_nets.append(
#                 MLP(config.in_channels, config.hidden_channels, config.hidden_channels, config.num_layers, config.dropout)
#             )
#             self.node_nets[-1].reset_parameters()

#         self.proj = torch.nn.Linear(config.hidden_channels, config.n_freqs * 2 + 1)
#         self.n_freqs = config.n_freqs
#         self.bn = nn.BatchNorm1d(config.in_channels)

#     def reset_parameters(self):
#         for net in self.node_nets:
#             net.reset_parameters()
#         self.proj.reset_parameters()

#     def get_input_struct(self, batch) -> ParamDict:
#         batch = ParamDict(batch)
#         return ParamDict(x=batch.x)

#     def get_output_struct(self, batch) -> ParamDict:
#         batch = ParamDict(batch)
#         return ParamDict(vac_real=batch.vac_real, vac_imag=batch.vac_imag, vdc=batch.vdc)

#     def get_node_features(self, inputs: ParamDict) -> torch.Tensor:
#         node_embs = [F.relu(net(self.bn(inputs.x))) for net in self.node_nets]
#         node_embs = torch.stack(node_embs, 1)
#         return node_embs

#     def project_node_to_output(self, inputs: ParamDict) -> ParamDict:
#         projected_nodes = [self.proj(inputs.node_embs[:, i, :]) for i in range(inputs.node_embs.shape[1])]
#         projected_nodes = torch.stack(projected_nodes, 1)

#         output = ParamDict(
#             node_embs=inputs.node_embs,
#             vac_real=projected_nodes[..., :self.n_freqs],
#             vac_imag=projected_nodes[..., self.n_freqs:2*self.n_freqs],
#             vdc=projected_nodes[..., -1:]
#         )

#         return output
