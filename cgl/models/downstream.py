from typing import Dict

import torch
from torch import optim
import torch.nn as nn
from torch.optim import Adam

from cgl.utils.params import ParamDict
from cgl.utils.torch import EMA

import pytorch_lightning as pl
from enum import IntEnum, auto

class LossType(IntEnum):
    REGRESSION = auto()
    CLASSIFICATION = auto()


class CircuitPredictorBase(pl.LightningModule):

    def __init__(
        self, 
        config: ParamDict,
        node_emb_module: nn.Module,
        output_labels_dict = Dict[str, LossType],    
    ) -> None:
        super().__init__()

        config.setdefault('lin_output_head', True)
        self.save_hyperparameters('config')

        self.config = config
        self.node_emb_module = node_emb_module
        self.output_labels_dict = output_labels_dict
        
        self.reg_keys = [k for k in output_labels_dict if output_labels_dict[k] is LossType.REGRESSION]
        self.class_keys = [k for k in output_labels_dict if output_labels_dict[k] is LossType.CLASSIFICATION]

        if self.config.readout == 'mlp':
            no_features = node_emb_module.config.hidden_channels * node_emb_module.config.num_nodes
        else:
            no_features = node_emb_module.config.hidden_channels 

        if self.config.lin_output_head:
            self.output_head = nn.Linear(no_features, len(output_labels_dict))
        else:
            hsize = 512
            self.output_head = nn.Sequential(
                nn.Linear(no_features, hsize),
                nn.ReLU(),
                nn.Linear(hsize, hsize),
                nn.ReLU(),
                nn.Linear(hsize, hsize),
                nn.ReLU(),
                nn.Linear(hsize, len(output_labels_dict))
            )

        # 0.95 was derived based on trial and error
        self.val_ema = EMA(0.95)

    def reset_parameters(self):
        if isinstance(self.output_head, nn.Sequential):
            for mod in self.output_head:
                if isinstance(mod, nn.Linear):
                    mod.reset_parameters()
        else:
            self.output_head.reset_parameters()

    def configure_optimizers(self):
        optimizer_dict = {}
        if self.config.train_node_emb:
            parameters = self.parameters()
        else:
            parameters = self.output_head.parameters()

        optimizer_cls = self.config.get('optimizer', 'Adam')
        optimizer_params = self.config.get('optimizer_params', {})
        if optimizer_cls == 'Adam':
            optimizer = optim.Adam(parameters, lr=self.config.lr, **optimizer_params)
        elif optimizer_cls == 'SGD':
            optimizer = optim.SGD(parameters, lr=self.config.lr, **optimizer_params)
        elif optimizer_cls == 'Adagrad':
            optimizer = optim.Adagrad(parameters, lr=self.config.lr, **optimizer_params)
        optimizer_dict.update(optimizer=optimizer)

        scheduler = self.config.get('lr_schedule', None)
        schedule_params = self.config.get('lr_scheduler_params', {})
        if scheduler is not None:
            if scheduler == 'ReduceLROnPlateau':
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **schedule_params)
                optimizer_dict.update(monitor='valid_loss_total')
            optimizer_dict.update(lr_scheduler=scheduler)
        return optimizer_dict

    
    def forward(self, inputs) -> ParamDict:
        raise NotImplementedError

    def get_input_struct(self, batch):
        raise NotImplementedError
    
    def get_graph_emb(self, node_embs):
        if self.config.readout == 'avg_pool':
            return node_embs.mean(1)
        elif self.config.readout == 'max_pool':
            return node_embs.max(1)
        elif self.config.readout == 'mlp':
            return node_embs.reshape(-1, node_embs.shape[-1] * node_embs.shape[-2])
        else:
            raise ValueError(f'Unknown readout type {self.config.readout}')

    def get_output_struct(self, batch):
        # return ParamDict(**{k: v for k, v in batch.items() if k in self.output_labels_dict})
        return ParamDict(**{k: batch[k] for k in self.output_labels_dict})

    def get_loss(self, pred_output: ParamDict, true_output: ParamDict):
        loss_dict = ParamDict()
        total = 0.
        for key in self.reg_keys:
            dtype = pred_output[key].dtype
            # in mlp dataset the true output has shape of (batch, 1)
            loss = nn.MSELoss()(pred_output[key], true_output[key].type(dtype).squeeze(-1))
            loss_dict[key] = loss
            total += loss

        for key in self.class_keys:
            dtype = pred_output[key].dtype
            # in mlp dataset the true output has shape of (batch, 1)
            loss = nn.BCEWithLogitsLoss()(pred_output[key], true_output[key].type(dtype).squeeze(-1))
            loss_dict[key] = loss
            total += loss

        total /= len(self.output_labels_dict)
        loss_dict.update(total=total)
        return loss_dict

    def training_step(self, batch, batch_idx):
        res = self._compute_ff(batch)
        self._log_dict_with_prefix(res.loss, 'train_loss')
        return res.loss.total

    def validation_step(self, batch, batch_idx):
        res = self._compute_ff(batch)
        self._log_dict_with_prefix(res.loss, 'valid_loss')
        return res.loss.total

    def validation_epoch_end(self, outputs) -> None:
        # this val_loss_ema is used for monitoring for checkpoints and to avoid noisy validation loss
        # we do this in epoch end because we want to initialize it to epoch's validation not batch validation in step zero
        val_loss_ema = self.val_ema(torch.stack(outputs).mean())
        self.log('valid_loss_total_ema', val_loss_ema)

    def test_step(self, batch, batch_idx):
        res = self._compute_ff(batch)
        self.log_dict(res.loss)

    def predict(self, batch, compute_loss=False):
        self.eval()
        with torch.no_grad():
            return self._compute_ff(batch, compute_loss)
    
    def _compute_ff(self, batch, compute_loss=True):
        input_struct = self.get_input_struct(batch)
        output_struct = self(input_struct)
        if compute_loss:
            true_output_struct = self.get_output_struct(batch)
            loss = self.get_loss(output_struct, true_output_struct)
            return ParamDict(input=input_struct, output=output_struct, loss=loss)
        return ParamDict(input=input_struct, output=output_struct)

    def _log_dict_with_prefix(self, dictionary, prefix='', **kwargs):
        if not prefix:
            self.log_dict(dictionary, **kwargs)
        else:
            for key in dictionary:
                self.log(f'{prefix}_{key}', dictionary[key], **kwargs)


class CircuitPredictorMLPBackbone(CircuitPredictorBase):

    def forward(self, inputs) -> ParamDict:
        node_embs = self.node_emb_module.get_node_features(inputs)

        # # debugging
        # node_embs_flat = node_embs.reshape(-1, node_embs.shape[-1] * node_embs.shape[-2])
        # pca = PCA(n_components=min(node_embs_flat.shape))
        # pca.fit(node_embs_flat.detach().cpu().numpy())
        # exp_ratio = np.cumsum(pca.explained_variance_ratio_)
        # print(exp_ratio[:10])
        # breakpoint()
        
        graph_emb = self.get_graph_emb(node_embs)
        output_score = self.output_head(graph_emb)
        ff_dict = ParamDict(node_embs=node_embs, graph_emb=graph_emb)
        for i, key in enumerate(self.output_labels_dict):
            ff_dict[f'{key}'] = output_score[:, i]
        return ff_dict

    def get_input_struct(self, batch):
        return ParamDict(x=batch['x'])


import numpy as np
from sklearn.decomposition import PCA
class CircuitPredictorGNNBackbone(CircuitPredictorBase):

    def forward(self, inputs) -> ParamDict:
        # TODO: not sure if this is the best way to infer batch size from the Data object
        bsize = len(inputs.data.ptr) - 1
        node_embs = self.node_emb_module.get_node_features(inputs)
        node_embs = node_embs.reshape(bsize, -1, node_embs.shape[-1])

        # # debugging
        # node_embs_flat = node_embs.reshape(-1, node_embs.shape[-1] * node_embs.shape[-2])
        # pca = PCA(n_components=min(node_embs_flat.shape))
        # pca.fit(node_embs_flat.detach().cpu().numpy())
        # exp_ratio = np.cumsum(pca.explained_variance_ratio_)
        # print(exp_ratio[:10])
        # breakpoint()

        graph_emb = self.get_graph_emb(node_embs)
        output_score = self.output_head(graph_emb)
        ff_dict = ParamDict(node_embs=node_embs, graph_emb=graph_emb)
        for i, key in enumerate(self.output_labels_dict):
            ff_dict[f'{key}'] = output_score[:, i]
        return ff_dict

    def get_input_struct(self, batch):
        batch.x = torch.cat([batch.x, batch.type_tens], -1).to(self.device)
        return ParamDict(data=batch)

