"""
The train and test datasets should be handled separately because of their different pkl structure
"""

import numpy as np
from pathlib import Path
import os

import torch
from torch.utils.data import Dataset, DataLoader

from cgl.utils.file import read_pickle
from cgl.utils.pdb import register_pdb_hook
from cgl.bagnet_gnn.dict2graph import Dict2Graph

register_pdb_hook()

MAX_SPECS = ('gain', 'ugbw', 'pm', 'psrr', 'cmrr')
MIN_SPECS = ('tset', 'offset_sys', 'ibias')

TORCH_MEAN = torch.tensor([6.0134e-12, 5.3201e+01, 4.9209e+01, 3.8199e+01, 6.2449e+01, 3.7848e+01,
                           6.8371e+01, 5.9423e+02])
TORCH_STD = torch.tensor([2.1734e-12, 2.3379e+01, 2.3961e+01, 2.5193e+01, 2.5965e+01, 2.3892e+01,
                          2.4936e+01, 2.6647e+02])


class InputHandler:

    def __init__(self, meta_data_path='', is_graph=False) -> None:
        
        if not meta_data_path:
            meta_data_path = os.environ['META_DATA_PATH']
            
        self.is_graph = is_graph

        self.dict2graph = None
        if is_graph:
            self.dict2graph = Dict2Graph(meta_data_path)

    def _get_torch_repr(self, data_dict):
        params = data_dict['params']
        vec = torch.tensor([params[key] for key in sorted(params.keys())], dtype=torch.float)
        norm_vec = (vec - TORCH_MEAN) / TORCH_STD
        return norm_vec

    def _get_graph_repr(self, data_dict):
        params = data_dict['params']
        graph = self.dict2graph.param_to_graph(params)
        return graph

    def get_input_repr(self, data_dict):
        if self.is_graph:
            return self._get_graph_repr(data_dict)
        return  self._get_torch_repr(data_dict)


class BagNetDataset(Dataset):


    def __init__(self, *, datapath='', is_graph=False) -> None:
        super().__init__()

        self.input_handler = InputHandler(datapath, is_graph)
        self.is_graph = self.input_handler.is_graph


    def _get_labels(self, input_a_dict, input_b_dict):
        
        spec_a = input_a_dict['specs']
        spec_b = input_b_dict['specs']

        labels = {}
        for key in MAX_SPECS:
            labels[key] = torch.tensor(spec_a[key] > spec_b[key], dtype=torch.long)
        for key in MIN_SPECS:
            labels[key] = torch.tensor(spec_a[key] < spec_b[key], dtype=torch.long)

        return labels


    def _get_comparison_data(self, sample_a, sample_b):
        
        input_a = self.input_handler.get_input_repr(sample_a)
        input_b = self.input_handler.get_input_repr(sample_b)

        labels = self._get_labels(sample_a, sample_b)

        return dict(input_a=input_a, input_b=input_b, **labels)


class BagNetDatasetTrain(BagNetDataset):

    def __init__(self, datapath, optim_round=0, is_graph=False) -> None:
        super().__init__(datapath, is_graph=is_graph)

        self.datafile = Path(datapath) / 'train.pkl'
        self.data_all = read_pickle(self.datafile)
        self.round_max = len(self.data_all) - 1

        self.optim_round = optim_round
        assert 0 <= optim_round <= self.round_max

        self.data_cur_round = []
        for round_idx in range(self.optim_round + 1):
            self.data_cur_round += self.data_all[round_idx]
    
    
    def __len__(self):
        n = len(self.data_cur_round)
        return n * (n-1) // 2 # all pair combinations of the dataset

    def _get_paired_idx(self):
        # imagine an upper triangular matric of pairs
        # select row r = rand(0, n-2), excluding r_{n-1}
        # then select the col = rand(r+1, n)
        n = len(self.data_cur_round)
        idx_a = np.random.randint(n - 1)
        idx_b = np.random.randint(idx_a + 1, n)

        if np.random.rand() < 0.5:
            idx_a, idx_b = idx_b, idx_a

        return idx_a, idx_b

    def __getitem__(self, idx):

        idx_a, idx_b = self._get_paired_idx()

        sample_a = self.data_cur_round[idx_a]
        sample_b = self.data_cur_round[idx_b]

        output = self._get_comparison_data(sample_a, sample_b)
        return output


class BagNetDatasetTest(BagNetDataset):

    def __init__(self, datapath, is_graph=False) -> None:
        super().__init__(datapath, is_graph=is_graph)
        self.datafile = Path(datapath) / 'test.pkl'
        self.data_all = read_pickle(self.datafile)

        self.data_merged = []
        for round_idx in self.data_all.keys():
            self.data_merged += self.data_all[round_idx]

    def __len__(self):
        return len(self.data_merged)

    def __getitem__(self, idx):
        input_dict = self.data_merged[idx]

        sample_a = input_dict['input1']
        sample_b = input_dict['input2']

        output = self._get_comparison_data(sample_a, sample_b)
        return output


class BagNetOnlineDataset(BagNetDataset):

    def __init__(self, input_a, input_b, labels, is_graph=False) -> None:
        super().__init__(is_graph=is_graph)

        self.input_a = input_a
        self.input_b = input_b
        self.labels = labels

    def __len__(self):
        return len(self.input_a)

    
    def __getitem__(self, index):

        sample_a = dict(params=self.input_a[index])
        sample_b = dict(params=self.input_b[index])

        input_a = self.input_handler.get_input_repr(sample_a)
        input_b = self.input_handler.get_input_repr(sample_b)
        label = {k: torch.tensor(v[index]).long() for k, v in self.labels.items()}

        return dict(input_a=input_a, input_b=input_b, **label)
    

if __name__ == '__main__': 

    # from torch.utils.data import DataLoader

    # dset_0 = DatasetTrain('datasets/bagnet_gnn', optim_round=0)
    # dset_15 = BagNetDatasetTrain('datasets/bagnet_gnn', optim_round=15)

    # dloader = DataLoader(dset_15, batch_size=len(dset_15))
    # batch = next(iter(dloader))

    # print('mean')
    # print(batch['input_a'].mean(0))
    # print('std')
    # print(batch['input_a'].std(0))

    # tset = BagNetDatasetTest('datasets/bagnet_gnn')
    # tset[0]
    # breakpoint()

    train_set = BagNetDatasetTrain('datasets/bagnet_gnn', optim_round=15)
    test_set = BagNetDatasetTest('datasets/bagnet_gnn')


    train_x = []
    for idx in range(len(train_set)):
        data = train_set[idx]
        train_x.append(torch.cat([data['input_a'], data['input_b']], -1))
    train_x = torch.stack(train_x, 0).detach().cpu().numpy()

    test_x = []
    for idx in range(len(test_set)):
        data = test_set[idx]
        test_x.append(torch.cat([data['input_a'], data['input_b']], -1))
    test_x = torch.stack(test_x, 0).detach().cpu().numpy()

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    print('Running TSNE for train ...')
    train_2d = TSNE(n_components=2, verbose=True).fit_transform(train_x)
    print('Running TSNE for test ...')
    test_2d = TSNE(n_components=2, verbose=True).fit_transform(test_x)

    plt.scatter(train_2d[:, 0], train_2d[:, 1], color='red', s=5)
    plt.scatter(test_2d[:, 0], test_2d[:, 1], color='blue', s=5)
    plt.savefig('tsne_test_train.png')

    print(train_set[0])
    print('-'*30)
    print(test_set[0])
    breakpoint()