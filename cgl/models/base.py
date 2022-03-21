from collections import defaultdict
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl

from cgl.utils.params import ParamDict
from cgl.utils.torch import EMA
from cgl.eval.evaluator import NodeEvaluator

class Huberloss(nn.Module):

    def __init__(self, delta=1.0, reduction='mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, y1, y2):
        mask = (y1 - y2).abs() < self.delta 
        loss1 = 0.5 * (y1 - y2) ** 2
        loss2 = self.delta * ((y1 - y2).abs() - 0.5 * self.delta)
        loss = mask * loss1 + (~mask) * loss2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f'reduction keyword {self.reduction} is not valid')
        


class BaseNodeEmbeddingModule(pl.LightningModule):

    def __init__(self, config: ParamDict) -> None:
        super().__init__()
        config.setdefault('cosine_scheduler', False)
        config.setdefault('loss_type', 'mse')
        self.save_hyperparameters()
        self.output_labels = config.output_labels # {'vdc': 1, 'vac_mag': 101, 'vac_ph': 101}
        self.config = config
        self.val_ema = EMA(0.99)
        self.evaluator = NodeEvaluator(config.bins)
        self.build_network(config)

    # def on_fit_start(self) -> None:
    #     pl.seed_everything(self.hparams['seed'])

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config.lr)
        return optimizer

    def build_network(self, config):
        raise NotImplementedError
        
    def reset_parameters(self):
        raise NotImplementedError
    
    def get_input_struct(self, batch) -> ParamDict:
        # digests batch data to a digestable format for get_node_features
        raise NotImplementedError

    def get_output_struct(self, batch) -> ParamDict:
        # extracts the true output data from the batch 
        raise NotImplementedError

    def get_node_features(self, inputs: ParamDict) -> torch.Tensor:
        # converts inputs to node features
        raise NotImplementedError

    def project_node_to_output(self, inputs: ParamDict) -> ParamDict:
        # converts node embeddings to the predicted output
        raise NotImplementedError

    def get_loss(self, pred_output: ParamDict, true_output: ParamDict):
        for key in true_output:
            true_output[key] = true_output[key].type(pred_output[key].dtype)

        total = 0.0
        loss_dict, eval_dict = ParamDict(), ParamDict()
        for label in self.output_labels:
            ytrue = true_output[label]
            ypred = pred_output[label]
            non_nan_inds = ~torch.isnan(ytrue)

            if self.config.loss_type == 'mse':
                loss = nn.MSELoss()(ytrue[non_nan_inds], ypred[non_nan_inds])
            elif self.config.loss_type == 'l1':
                loss = nn.L1Loss()(ytrue[non_nan_inds], ypred[non_nan_inds])
            elif self.config.loss_type == 'huber':
                loss = Huberloss()(ytrue[non_nan_inds], ypred[non_nan_inds])
            loss_dict[label] = loss
            eval_dict[f'{label}_acc'] = self.evaluator.eval(dict(y_true=ytrue[non_nan_inds], y_pred=ypred[non_nan_inds]))
            total += loss

        loss_dict['total'] = total
        return loss_dict, eval_dict

    def forward(self, inputs: ParamDict) -> ParamDict:
        node_embs = self.get_node_features(inputs)
        inputs.update(node_embs=node_embs)
        return self.project_node_to_output(inputs)

    def training_step(self, batch, batch_idx):
        if self.config.get('finetune', False):
            # batch norm and layer norm should not compute moving averages
            # print('model is in eval mode')
            self.eval()

        res = self._compute_ff(batch)
        # self._log_dict_with_prefix(res.loss, 'train_loss')
        # self._log_dict_with_prefix(res.eval, 'train')
        # return dict(loss=res.loss.total, eval=res.eval, output=res.output)
        # if self.global_step == 10:
        #     params = dict(self.named_parameters())
        #     param_name = 'proj.nets.8.weight'
        #     noise = 0.01 * torch.rand_like(params[param_name])
        #     params[param_name].data = noise
        summary = {k: v for k, v in res.items() if k not in ('input', 'output')}
        output = dict(loss=res.loss.total, summary=summary)
        return output

    def validation_step(self, batch, batch_idx):
        res = self._compute_ff(batch)
        # self._log_dict_with_prefix(res.loss, 'valid_loss', on_epoch=True)
        # self._log_dict_with_prefix(res.eval, 'valid')
        summary = {k: v for k, v in res.items() if k not in ('input', 'output')}
        output = dict(loss=res.loss.total, summary=summary)
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
        input_struct = self.get_input_struct(batch)
        pred_output_struct = self(input_struct)
        if compute_loss:
            true_output_struct = self.get_output_struct(batch)
            loss_dict, eval_dict = self.get_loss(pred_output_struct, true_output_struct)
            return ParamDict(input=input_struct, output=pred_output_struct, loss=loss_dict, eval=eval_dict)
        return ParamDict(input=input_struct, output=pred_output_struct)

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