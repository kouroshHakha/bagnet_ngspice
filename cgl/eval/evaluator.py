import numpy as np
from sklearn.metrics import roc_auc_score

try:
    import torch
except ImportError:
    torch = None

class Evaluator:

    def __init__(self, metric='mse', **kwargs) -> None:
        self.eval_metric = metric

    def eval(self, input_dict):
        y_true, y_pred = self._parse_and_check_input(input_dict)
        if self.eval_metric == 'mse':
            return self._eval_mse(y_true, y_pred)
        elif self.eval_metric == 'rocauc':
            # y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_rocauc(y_true, y_pred)
        elif self.eval_metric == 'iou':
            return self._eval_iou(y_true, y_pred)
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

    def _parse_and_check_input(self, input_dict):
        raise NotImplementedError

    def _eval_mse(self, y_true, y_pred):
        raise NotImplementedError

    def _eval_rocauc(self, y_true, y_pred):
        raise NotImplementedError

    def _eval_iou(self, y_true, y_pred):
        raise NotADirectoryError


class NodeEvaluator:

    def __init__(self, bins=1000) -> None:
        self.bins = bins

    # TODO: Does not support node classification yet.
    def _parse_and_check_input(self, input_dict):
        if not 'y_true' in input_dict:
            raise RuntimeError('Missing key of y_true')
        if not 'y_pred' in input_dict:
            raise RuntimeError('Missing key of y_pred')

        y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

        # converting to torch.Tensor to numpy on cpu
        if torch is not None and isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()

        if torch is not None and isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        ## check type
        if not isinstance(y_true, np.ndarray):
            raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

        if not y_true.shape == y_pred.shape:
            raise RuntimeError('Shape of y_true and y_pred must be the same')


        return y_true, y_pred


    def eval(self, input_dict, return_cond=False):
        
        # it ignores nans
        y_true, y_pred = self._parse_and_check_input(input_dict)
        threshold = (np.nanmax(y_true) - np.nanmin(y_true)) / self.bins
        cond = np.abs(y_pred - y_true) < threshold
        if return_cond:
            return cond
        return cond.sum() / cond.flatten().shape[0]


class NodeClassEvaluator(NodeEvaluator):

    
    # TODO: Does not support node classification yet.
    def _parse_and_check_input(self, input_dict):
        if not 'y_true' in input_dict:
            raise RuntimeError('Missing key of y_true')
        if not 'y_pred' in input_dict:
            raise RuntimeError('Missing key of y_pred')

        y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

        # converting to torch.Tensor to numpy on cpu
        if torch is not None and isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()

        if torch is not None and isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        ## check type
        if not isinstance(y_true, np.ndarray):
            raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

        return y_true, y_pred

    def eval(self, input_dict, return_cond=False):
        y_true, logit_pred = self._parse_and_check_input(input_dict)

        # assuming max and min are 1, 0: take the class x 0.01 + 1 / 2 / self.bins
        y_pred = np.argmax(logit_pred, -1) / self.bins + 1 / 2 / self.bins
        threshold = (np.nanmax(y_true) - np.nanmin(y_true)) / self.bins
        cond = np.abs(y_pred - y_true) < threshold
        if return_cond:
            return cond
        return cond.sum() / cond.flatten().shape[0]

class GraphEvaluator(Evaluator):

    
    def _parse_and_check_input(self, input_dict):
        if self.eval_metric in ['rocauc', 'mse']:
            if not 'y_true' in input_dict:
                raise RuntimeError('Missing key of y_true')
            if not 'y_pred' in input_dict:
                raise RuntimeError('Missing key of y_pred')

            y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

            '''
                y_true: numpy ndarray or torch tensor of shape (num_graphs, num_tasks)
                y_pred: numpy ndarray or torch tensor of shape (num_graphs, num_tasks)
            '''

            # converting to torch.Tensor to numpy on cpu
            if torch is not None and isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()

            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()

            ## check type
            if not (isinstance(y_true, np.ndarray) and isinstance(y_true, np.ndarray)):
                raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

            if not y_true.shape == y_pred.shape:
                raise RuntimeError('Shape of y_true and y_pred must be the same')

            if not y_true.ndim == 2:
                raise RuntimeError('y_true and y_pred must to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

            return y_true, y_pred

        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))


    def _eval_rocauc(self, y_true, y_pred):
        '''
            compute ROC-AUC and AP score averaged across tasks
        '''

        rocauc_list = []

        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                is_labeled = y_true[:,i] == y_true[:,i]
                rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i]))

        if len(rocauc_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

        return {'rocauc': sum(rocauc_list)/len(rocauc_list), 'rocauc_list': rocauc_list}


    def _eval_mse(self, y_true, y_pred):
        '''
            mse: numpy ndarray of shape (num_tasks), Mean square error per task
        '''
        
        mse = ((y_true - y_pred) ** 2).mean(0)
        return dict(mse_per_task=mse, mse=mse.mean())