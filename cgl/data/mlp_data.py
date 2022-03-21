
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data.dataset import Dataset

from cgl.data.graph_data import CircuitGraphDataset
from cgl.utils.params import ParamDict


class CircuitMLPDataset(Dataset):

    def __init__(self, root, split='train', is_downstream=False) -> None:
        # in memory dataset
        # all the data is processed at once and then splitted into train/valid/test for the main task
        # for the downstream task the separate dataset is used.
        super().__init__()
        self.root = root
        self.is_downstream = is_downstream
        
        if is_downstream:
            self.output_keys = [
                'cost', 'cost_label',
                'gain', 'gain_label',
                'ugbw', 'ugbw_label',
                'tset', 'tset_label',
                'psrr', 'psrr_label',
                'ibias', 'ibias_label',
                'offset_sys', 'offset_sys_label',
            ]
        else:
            self.output_keys = ['vac_imag', 'vac_real', 'vdc']

        self.processed_dir = Path(root) / ('test' if is_downstream else 'train') / 'processed'
        self.split = None

        all_exists = all([(self.processed_dir / pfile).exists() for pfile in self.processed_file_names])
        if not all_exists:
            self.process()

        self.data = ParamDict(torch.load(self.processed_dir / 'mlp_data.pt', map_location='cpu'))
        self.stats = ParamDict(torch.load(self.processed_dir / 'mlp_stats.pt', map_location='cpu'))
        self.splits = torch.load(self.processed_dir / 'splits.pt', map_location='cpu')[split]
        
        # sanity check: just keep the ids that end with _0
        if is_downstream:
            self.splits = self.splits[::9]
        
        self.split_data = {k: v[self.splits] for k, v in self.data.items() if k != 'freq'}
        if not self.is_downstream:
            self.split_data['freq'] = self.data['freq']

    @property
    def processed_file_names(self):
        return ['mlp_data.pt', 'mlp_stats.pt']

    def process(self):
        gdataset = CircuitGraphDataset(self.root, mode='test' if self.is_downstream else 'train')

        xlist = []
        output_list = {k: [] for k in self.output_keys}

        if not (self.processed_dir / 'mlp_stats.pt').exists() or (self.processed_dir / 'mlp_data.pt').exists():
            for data in tqdm(gdataset):
                xlist.append(data.circuit_params)
                for key in output_list:
                    output_list[key].append(data[key])
            
            x = torch.stack(xlist, 0)
            xmean, xstd = x.mean(0), x.std(0)
            xnorm = (x - xmean) / xstd
            xnorm[torch.isnan(xnorm) | torch.isinf(xnorm)] = 0.
            stats = ParamDict(mean=xmean, std=xstd)
            torch.save(dict(stats), self.processed_dir / 'mlp_stats.pt')

            data = ParamDict(x=xnorm)
            data.update(**{key: torch.stack(output_list[key], 0) for key in self.output_keys})
            if not self.is_downstream:
                data.update(freq=gdataset[0].freq)

            torch.save(dict(data), self.processed_dir / 'mlp_data.pt')

        # # inherit split behavior from graph dataset to make comparison possible
        # if not (self.processed_dir / 'mlp_split.pt').exists():
        #     split_dict = gdataset.splits
        #     torch.save(split_dict, self.processed_dir / 'mlp_split.pt')

    def __len__(self):
        return len(self.splits)

    def __getitem__(self, index):
        out = ParamDict(**{k: v[index] for k, v in self.split_data.items() if k != 'freq'})
        if not self.is_downstream:
            out.freq = self.split_data['freq']
        return out