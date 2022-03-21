import time
from typing import Tuple, Dict
import warnings

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset, InMemoryDataset
from torch_geometric.utils import dense_to_sparse


from pathlib import Path
import os.path as osp
import json
import numpy as np
from tqdm import tqdm

from cgl.utils.trie import Trie
from cgl.utils.params import ParamDict
from cgl.utils.file import write_hdf5, read_hdf5, read_yaml, write_yaml, HDF5Error
from cgl.utils.pdb import register_pdb_hook
register_pdb_hook()

import time


class CircuitGraphDataset(Dataset):

    FILE_ID = {
        'opamp_pt': '1Qv4XQVh8Bp_MwYPpuOG5vBEJ0EAHOjlF',
        'opamp_biased_pmos': '102h9nueJt9zBMkwWw_WcJmkQRw9rz7Xa',
    }
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, mode='train', circuit_type='opamp_pt'):

        # self.feat_dim = len(self.node_values) + len(self.node_types)
        self.mode = mode
        self._stats = None
        self.circuit_type = circuit_type

        # This Trie is used for node_type encoding that will be done later
        # The node type key words are hierarchically added to a tree strcuture 
        # so that they can be assigned hierarchical one-hot encoding later
        self.node_type_trie = Trie()

        super().__init__(
            root=str(Path(root) / mode), 
            transform=transform, pre_transform=pre_transform, pre_filter=pre_filter
        )

        # in-memory attributes to map back to the original graph 
        info = torch.load(Path(self.processed_dir) / 'graphs.pt')
        self.graph_ids = info['graph_ids']
        self.graph_nodes = info['graph_nodes']

        # load the trie even if the process() is not executed
        self._stats = read_yaml(Path(self.processed_dir) / 'graph_stats.yaml')
        self.node_type_trie = Trie.load(Path(self.processed_dir) / 'node_type.trie')


        split_path = Path(self.processed_dir) / 'splits.pt'
        if self.mode == 'train':
            if split_path.exists():
                self.splits = torch.load(split_path, map_location='cpu')
            else:
                # 1000 test and valid datasets, the rest is for training
                # if self.len() < 3000:
                #     raise ValueError('This split procedure needs more than 3000 data points')
                inds = torch.randperm(self.len())
                # valid_idx = int(len(inds) * 1/3)
                # test_idx = int(len(inds) * 2/3)
                self.splits = dict(train=inds[:-2000], valid=inds[-2000:-1000], test=inds[-1000:])
                torch.save(self.splits, Path(self.processed_dir) / 'splits.pt')
        else:
            inds = torch.arange(self.len())
            self.splits = dict(train=inds[:800*9], valid=inds[800*9:900*9], test=inds[900*9:])
            torch.save(self.splits, Path(self.processed_dir) / 'splits.pt')


    @property
    def raw_file_names(self):
        # A list of files in the raw_dir which needs to be found in order to skip the download.
        return [f'{self.mode}.json']

    @property
    def processed_file_names(self):
        return ['graph_stats.yaml', 'graphs.pt', 'node_type.trie']

    @property
    def stats(self):
        if self._stats is None:
            self._stats = read_yaml(Path(self.processed_dir) / 'graph_stats.yaml')
        return self._stats

    def download(self):
        # Downloads raw data into raw_dir
        from cgl.utils.download import download_file_from_google_drive, decompress
        import tempfile

        # remove raw_dir cause it will get created again. 
        # For some reason gometric_pytorch creates it once before coming to download function.
        Path(self.raw_dir).rmdir()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tar_file = str(Path(tmp_dir) / 'raw.tar.gz')
            print('Downloading the dataset ...')
            download_file_from_google_drive(self.FILE_ID[self.circuit_type],  tar_file)
            print(f'Unzipping {tar_file} to {self.raw_dir} ...')
            decompress(tar_file, str(Path(self.raw_dir).parent))
        print('Download and Extraction complete.')

    def process(self):
        # Processes raw data and saves it into the processed_dir
        # _tmp are the versions that were generated using the modify_content fn below
        # without _tmp are the ones that were originally generated early in the project.
        s = time.time()

        if not (Path(self.raw_dir) / f'{self.mode}_processed.json').exists():
            with open(Path(self.raw_dir) / f'{self.mode}.json', 'r') as f:
                print(f'Reading the {self.mode} json file ...')
                content = json.load(f)
                print(f'Read successful in {time.time() - s:.2f} seconds!')

            content = modify_content(content)

            with open(Path(self.raw_dir) / f'{self.mode}_processed.json', 'w') as f:
                print('Saving the json file ...')
                json.dump(content, f)
                print('Saving successful!')
        else:
            with open(Path(self.raw_dir) / f'{self.mode}_processed.json', 'r') as f:
                print(f'Reading the {self.mode} json file ...')
                content = json.load(f)
                print(f'Read successful in {time.time() - s:.2f} seconds!')
            
        stats_fpath = Path(self.processed_dir) / 'graph_stats.yaml'
        trie_fpath = Path(self.processed_dir) / 'node_type.trie'
        # if trie path doesn't exist but stats exists we should run the get_stats() one more time, 
        # b/c trie is created during that process.
        if stats_fpath.exists() and trie_fpath.exists():
            s = time.time()
            print('Reading the stats file ...')
            self._stats = read_yaml(stats_fpath)
            self.node_type_trie = Trie.load(trie_fpath)
            print(f'Read was successful in {time.time() - s:.2f} seconds!')
        else:
            self._stats = self.get_stats(content)
            s = time.time()
            print('Saving the stats file ...')
            write_yaml(stats_fpath, self._stats)
            print(f'Saving was successful in {time.time() - s:.2f} seconds!')


        graph_ids = []
        graph_nodes = {}
        i = 0
        print('Converting the train.json file to individual graph torch files for efficient loading ...')
        for graph_dict in tqdm(content):
            try:
                node_map, data = self.graph_to_data(graph_dict)
            except HDF5Error:
                continue

            graph_ids.append(graph_dict['id'])
            graph_nodes[graph_dict['id']] = node_map

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, Path(self.processed_dir) / f'data_{i}.pt')
            i += 1

        if len(graph_ids) < len(content):
            warnings.warn(f'{len(content) - len(graph_ids)} graphs were not created.')

        if graph_ids:
            torch.save(dict(graph_ids=graph_ids, graph_nodes=graph_nodes), Path(self.processed_dir) / 'graphs.pt')

        print('Conversion finished.')

    def get_stats(self, content):
        print('Computing the Stats across all the graphs ...')
        print('This may take a while')
        # collect data accross all the graphs
        arrs = dict(
            # <node_type>: Mapping[<props_key>: <np.ndarray.stats>]
            inputs={},
            # Mapping[<prop_names>: <np.ndarray.stats>]
            outputs={}
        )

        s = time.time()
        # only compute the stat based on the first 1000 graphs since the order is random
        content_stats = content[:1000]
        for graph_dict in tqdm(content_stats):
            sim_data, gout_data = None, None
            try:
                # TODO: for some reason hdf5 on small files is slow. This takes ~ 1hr
                if 'sim_fpath' in graph_dict:
                    sim_data = read_hdf5(Path(self.raw_dir) / graph_dict['sim_fpath'])
                if 'gout_fpath' in graph_dict:
                    # TODO: test data is left out for now
                    gout_data = read_hdf5(Path(self.raw_dir) / graph_dict['gout_fpath'])
            except OSError:
                continue

            nodes = graph_dict['nodes']
            for node in nodes.values():
                ntype = tuple(node['type'])
                nprops = node['props']
                self.node_type_trie.add(ntype)
                
                # first time seeing ntype
                if ntype not in arrs['inputs']:
                    arrs['inputs'][ntype] = {}
                    for key in nprops:
                        arrs['inputs'][ntype][key] = []
                
                for key in nprops:
                    arrs['inputs'][ntype][key].append(nprops[key])
            
            if sim_data:
                if not arrs['outputs']:
                    arrs['outputs'] = {k: [] for k in ['vac_real', 'vac_imag', 'vdc', 'iac_real', 'iac_imag', 'idc']}
                vnodes = sim_data['nodes']
                vsrcs = sim_data['vsrcs']
                arrs['outputs']['vac_real'] += [v['real'] for v in vnodes.values()]
                arrs['outputs']['vac_imag'] += [v['imag'] for v in vnodes.values()]
                arrs['outputs']['vdc'] += [v['dc'] for v in vnodes.values()]
                arrs['outputs']['iac_real'] += [i['real'] for i in vsrcs.values()]
                arrs['outputs']['iac_imag'] += [i['imag'] for i in vsrcs.values()]
                arrs['outputs']['idc'] += [i['dc'] for i in vsrcs.values()]

            if gout_data:
                if not arrs['outputs']:
                    arrs['outputs'] = {k: [] for k in gout_data}
                for k in gout_data:
                    arrs['outputs'][k] += [gout_data[k]]
                
        print(f'loop finished in {time.time() - s:.6f} seconds')

        def _combine_to_arrs(ptr_dict):
            for key, value in ptr_dict.items():
                if isinstance(value, list):
                    if isinstance(value[0], np.ndarray):
                        ptr_dict[key] = np.stack(value, 0)
                    else:
                        ptr_dict[key] = np.array(value)
                elif isinstance(value, dict):
                    _combine_to_arrs(value)

        def _get_stats(ptr_dict):
            stats_dict = {}
            for key, value in ptr_dict.items():
                if isinstance(value, dict):
                    stats_dict[key] = _get_stats(value)
                else:
                    stats_dict[key] = (float(value.mean()), float(value.std()))
            return stats_dict

        s = time.time()
        _combine_to_arrs(arrs)
        print(f'Arrays got combined in {time.time() - s:.6f} seconds!')

        # HACK: compute the ac voltages normalized to the output node (index 7 node of each graph)
        vac = (arrs['outputs']['vac_real'] + 1j * arrs['outputs']['vac_imag']).reshape(len(content_stats), -1, 101)
        vac_mag = np.abs(vac)
        # HACK: for downstream data we don't need to normalize since it's just the voltage gain from input to output anyways
        # we normalized specifically because of the current source in the pretraining data
        if self.circuit_type == 'opamp_pt':
            vac_mag_factor = vac_mag[:, 7].max(-1)[:, None, None] * np.ones_like(vac_mag)
            vac_mag = vac_mag / vac_mag_factor
        # stat will be the statistics of those nodes that have a non-zero magnitude
        vac_mag_cond = vac_mag > 0
        vac_mag_db = 20 * np.log(vac_mag[vac_mag_cond])
        vac_ph = np.angle(vac[vac_mag_cond])
        arrs['outputs']['vac_mag'] = vac_mag_db
        arrs['outputs']['vac_ph'] = vac_ph

        s = time.time()
        stats = _get_stats(arrs)
        print(f'Computed stats in {time.time() - s:.6f} seconds!')
        
        self.node_type_trie.save(Path(self.processed_dir) / 'node_type.trie')

        return stats

    def _process_outputs_from_sim_data(self, graph_dict, sim_data):
        vnodes = sim_data['nodes']
        i_vsrcs = sim_data['vsrcs']

        vac_r_list, vac_i_list, vdc_list = [], [], []

        freq = None
        for node_name, node_dict in graph_dict['nodes'].items():
            ntype = tuple(node_dict['type'])
            if ntype == ('VNode', 'NGND'):
                # extract outputs from these nodes
                vnode_key = node_name.split('_')[-1].lower()
                if vnodes:
                    node_sims = vnodes[vnode_key] 
                    if freq is None:
                        freq_vec = np.log(node_sims['freq'])
                        freq = (freq_vec - freq_vec.min()) / freq_vec.max() - 0.5
                    
                    vac_r_list.append(node_sims['real'])
                    vac_i_list.append(node_sims['imag'])
                    vdc_list.append(node_sims['dc'])

        vac_r_arr = np.stack(vac_r_list, 0)
        vac_i_arr = np.stack(vac_i_list, 0)
        vdc_arr = np.stack(vdc_list, 0)

        # HACK: normalize to output node (index 7): 
        # The reason for this is that when the input is current the voltage values will be large compared to others (because they are in ohms)
        # for downstream data we don't need to normalize since it's just the voltage gain from input to output anyways
        # we normalized specifically because of the current source in the pretraining data

        vac_arr = vac_r_arr + 1j * vac_i_arr
        vac_mag = np.abs(vac_arr)
        if self.circuit_type == 'opamp_pt':
            vac_mag_factor = vac_mag[7].max(-1)[None, None] * np.ones_like(vac_mag)
            vac_mag_norm = vac_mag / vac_mag_factor
        else:
            vac_mag_norm = vac_mag

        vac_mag = 20 * np.ma.log(np.abs(vac_mag_norm)).filled(np.nan)
        vac_ph = np.angle(vac_arr)
        vac_ph[np.isnan(vac_mag)] = np.nan

        vac_mag = normalize(vac_mag, self._stats['outputs']['vac_mag'])
        vac_ph = normalize(vac_ph, self._stats['outputs']['vac_ph'])
        vac_r_arr = normalize(vac_r_arr, self._stats['outputs']['vac_real'])
        vac_i_arr = normalize(vac_i_arr, self._stats['outputs']['vac_imag'])
        vdc_arr = normalize(vdc_arr, self._stats['outputs']['vdc'])[:, None]

        iac_r, iac_i, idc = [], [], []
        for edge in graph_dict['edges']:
            u, v = edge
            u_type = graph_dict['nodes'][u]['type']
            v_type = graph_dict['nodes'][v]['type']

            if u_type[0] == 'V' and v_type[0] == 'V':
                vsrc_key = u.split('_')[1]
                vsrc_sim = i_vsrcs.get(vsrc_key, None)

                if vsrc_sim: 
                    iac_r.append(normalize(vsrc_sim['real'], self._stats['outputs']['iac_real']))
                    iac_i.append(normalize(vsrc_sim['imag'], self._stats['outputs']['iac_imag']))
                    idc.append(normalize(vsrc_sim['dc'], self._stats['outputs']['idc']))
                else:
                    warnings.warn(f'I({vsrc_key}) is not found in sim data. Check the data generation stack.')

        
        outputs = ParamDict(
            freq = torch.tensor(freq),
            vac_mag = torch.tensor(vac_mag),
            vac_ph = torch.tensor(vac_ph),
            vac_real = torch.tensor(vac_r_arr),
            vac_imag = torch.tensor(vac_i_arr),
            vdc = torch.tensor(vdc_arr),
            iac_real = torch.tensor(np.stack(iac_r, 0)),
            iac_imag = torch.tensor(np.stack(iac_i, 0)),
            idc = torch.tensor(np.stack(idc, 0)),
        )

        return outputs

    def _process_outputs_from_downstream_data(self, graph_dict, ds_sim_data):
        regression_dictionary = {k: torch.tensor([normalize(ds_sim_data[k], self._stats['outputs'][k])]) for k in ds_sim_data}
        meet_spec_dictionary = {f'{k}_label': torch.tensor([v]) for k, v in meet_spec(ds_sim_data).items()}
        outputs = ParamDict(**regression_dictionary, **meet_spec_dictionary)
        return outputs


    def _process_structural_data_from_graph_dict(self, graph_dict):

        node_map = {}
        
        output_node_mask = torch.zeros(len(graph_dict['nodes']), dtype=torch.long)
        feature_list = []
        type_list = []
        type_enc_list = []
        enc_map = self.node_type_trie.get_leaf_encodings()
        
        for i, (node_name, node_dict) in enumerate(graph_dict['nodes'].items()):
            node_map[node_name] = i
            
            # encoding node properties into discretized feature vectors
            nprops = node_dict['props']
            ntype = tuple(node_dict['type'])

            if nprops:
                stat = self._stats['inputs'][ntype]
                node_x = []
                for value_key in nprops:
                    value_norm = float(normalize(nprops[value_key], stat[value_key]))
                    node_x.append(value_norm)
                
                node_x = torch.tensor(node_x)
            else:
                node_x = torch.tensor([])

            feature_list.append(node_x)
            type_list.append(ntype)
            type_enc_list.append(torch.tensor(enc_map[ntype]))
            output_node_mask[i] = ntype == ('VNode', 'NGND')

        output_current_plus = torch.zeros(len(graph_dict['nodes']), dtype=torch.long)
        output_current_minus = torch.zeros(len(graph_dict['nodes']), dtype=torch.long)
        edge_list = []
        for i, edge in enumerate(graph_dict['edges']):
            u, v = edge
            edge_list.append([node_map[u], node_map[v]])
            edge_list.append([node_map[v], node_map[u]])

            u_type = graph_dict['nodes'][u]['type']
            v_type = graph_dict['nodes'][v]['type']

            if u_type[0] == 'V' and v_type[0] == 'V':
                output_current_plus[node_map[u]] = 1
                output_current_minus[node_map[v]] = 1

        data = Data(edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous())
        data.x_list = feature_list
        data.x = torch.nn.utils.rnn.pad_sequence(feature_list, batch_first=True)
        data.type_list = type_list
        data.type_tens = torch.nn.utils.rnn.pad_sequence(type_enc_list, batch_first=True)
        circuit_params = graph_dict['circuit_params']
        data.circuit_params = torch.tensor(list(circuit_params.values()), dtype=torch.float64)

        data.output_node_mask = output_node_mask.bool()
        data.output_current_plus = output_current_plus.bool()
        data.output_current_minus = output_current_minus.bool()

        return data, node_map


    def graph_to_data(self, graph_dict) -> Tuple[Dict[str, int], Data]:
        """ 
        1. Create nodes using integer indices + encode their features
        2. Connect nodes using their edges
        3. Construct output arrays for nodes + output node masks (if available)
        4. Construct output arrays for edges + output edge masks (if available)
        """
        data, node_map = self._process_structural_data_from_graph_dict(graph_dict)
        sim_data, gout_data = None, None
        try:
            # TODO: for some reason hdf5 on small files is slow. This takes ~ 1hr
            if 'sim_fpath' in graph_dict:
                sim_data = read_hdf5(Path(self.raw_dir) / graph_dict['sim_fpath'])
            if 'gout_fpath' in graph_dict:
                gout_data = read_hdf5(Path(self.raw_dir) / graph_dict['gout_fpath'])
        except OSError:
            raise HDF5Error()
        
        if sim_data:
            update_dictionary = self._process_outputs_from_sim_data(graph_dict, sim_data)
            for k, v in update_dictionary.items():
                data.__setattr__(k, v)
        
        if gout_data:
            update_dictionary = self._process_outputs_from_downstream_data(graph_dict, gout_data)
            for k, v in update_dictionary.items():
                data.__setattr__(k, v)

        data.id = [graph_dict['id']]
        return node_map, data

    def len(self):
        return len(self.graph_nodes)

    def get(self, idx):
        # s = time.time()
        data = torch.load(Path(self.processed_dir) / f'data_{idx}.pt')
        # TODO: remove the list of tensors from the data object during process
        # This code makes sure that we are not passing list of tensors as it slows down the multi-process message passing. 
        modified_dict = {k: v for k, v in data.to_dict().items() if k not in ['x_list', 'type_list']}
        new_data = Data.from_dict(modified_dict)
        # print(f'load time: {time.time() - s}')
        return new_data

def normalize(arr, stat):
    # if the value is always constant std will be zero and you may as well make it zero
    if np.any(stat[1] == 0):
        return np.zeros_like(arr)
    else:
        return (arr - stat[0]) / stat[1]

def modify_content(content):
    '''Modifies the old x.json file format to the new one'''
    new_content = []
    for graph_dict in content:
        new_gdict = {'nodes': {}}
        nodes_list = graph_dict['nodes']
        circuit_params = {}
        for node in nodes_list:
            node_name = node['name']
            node_props = node['props']

            # get the term_type
            dev_class = node_props.get('device_class', None)
            if dev_class == 'M':
                dev_type = (dev_class, 'N' if node_props['is_nmos'] else 'P', node_props['terminal_type'])
            elif dev_class == 'I' or dev_class == 'V' or dev_class == 'R' or dev_class == 'C':
                dev_type = (dev_class, node_props['terminal_type'])
            elif dev_class is None:
                dev_type = ('VNode', 'GND' if node_props['is_gnd'] else 'NGND')
            else:
                raise ValueError(f'device class {dev_class} is not recognized.')

            # get the prop dict
            if dev_class == 'M':
                props = dict(w=node_props['w'])
            elif dev_class == 'I' or dev_class == 'V':
                props = dict(ac_mag=node_props['ac_mag'], ac_ph=node_props['ac_ph'], dc=node_props['dc'])
            elif dev_class == 'R' or dev_class == 'C':
                props = dict(value=node_props['value'])
            elif dev_class is None:
                props = {}
            else:
                raise ValueError(f'device class {dev_class} is not recognized.')

            # get graph level params
            if node['type'] == 'T':
                dev_key = node_name.split('_')[1]
                for prop_key in props:
                    param_key = f'{dev_key}_{prop_key}'
                    if param_key not in circuit_params:
                        circuit_params[param_key] = props[prop_key]
            

            new_gdict['nodes'][node_name] = dict(type=tuple(dev_type), props=props) 
        
        new_gdict.update(**{k: v for k,v in graph_dict.items() if k != 'nodes'}, circuit_params=circuit_params)
        new_content.append(new_gdict)

    return new_content


class CircuitInMemDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, mode='train', circuit_type='opamp_pt'):
        self.mode = mode
        self.circuit_type = circuit_type
        super().__init__(root, transform, pre_transform, pre_filter)
        # self.data, self.slices = torch.load(self.processed_paths[0])
        data = read_hdf5(Path(self.processed_dir) / 'data.h5')
        slices = read_hdf5(Path(self.processed_dir) / 'slices.h5')
        
        data_dict = {}
        for k in data:
            if k == 'id':
                data_dict[k] = data[k].astype('str').tolist()
            else:
                data_dict[k] = torch.from_numpy(data[k])
                
        self.data = Data.from_dict(data_dict)
        self.slices = {k: torch.from_numpy(slices[k]) for k in data}

        # NOTE: Be cautious of existing bugs here. Hacky version before deadline!!
        split_path = Path(self.processed_dir) / 'splits.pt'
        if split_path.exists():
            self.splits = torch.load(split_path, map_location='cpu')
        else:
            if self.mode == 'train':
                # 1000 test and valid datasets, the rest is for training
                # if self.len() < 3000:
                #     raise ValueError('This split procedure needs more than 3000 data points')
                inds = torch.randperm(self.len())
                # valid_idx = int(len(inds) * 1/3)
                # test_idx = int(len(inds) * 2/3)
                self.splits = dict(train=inds[:-2000], valid=inds[-2000:-1000], test=inds[-1000:])
                torch.save(self.splits, Path(self.processed_dir) / 'splits.pt')
            else:
                inds = torch.arange(self.len())
                self.splits = dict(train=inds[:800*9], valid=inds[800*9:900*9], test=inds[900*9:])
                torch.save(self.splits, Path(self.processed_dir) / 'splits.pt')

        # self.splits = torch.load(Path(self.processed_dir) / 'splits.pt', map_location='cpu')

        # mix valid and test datasets to remove the confounding factor of valid to test distribution shift
        torch.manual_seed(10)
        valid = torch.cat([self.splits['valid'][::2], self.splits['test'][::2]])
        test = torch.cat([self.splits['valid'][1::2], self.splits['test'][1::2]])
        valid = valid[torch.randperm(len(valid))]
        test = test[torch.randperm(len(test))]
        self.splits.update(valid=valid, test=test)

    @property
    def raw_dir(self):
        return osp.join(self.root, self.mode, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.mode, 'processed')

    @property
    def processed_file_names(self):
        return ['data.h5', 'slices.h5']

    def process(self):
        gdataset = CircuitGraphDataset(self.root, mode=self.mode, circuit_type=self.circuit_type)

        if not (Path(self.processed_dir) / 'data.h5').exists() or not (Path(self.processed_dir) / 'slices.h5').exists():
            data_list = []
            print('Creating in-memory dataset ...')
            for data in tqdm(gdataset):
                data_list.append(data)
            data, slices = self.collate(data_list)

            data_dict = {}
            for k in data.keys:
                if k == 'id':
                    data_dict[k] = np.array(data[k], dtype='S')
                else:
                    data_dict[k] = data[k].numpy()
            slices_dict = {k: slices[k].numpy() for k in slices.keys()}

            write_hdf5(data_dict, Path(self.processed_dir) / 'data.h5')
            write_hdf5(slices_dict, Path(self.processed_dir) / 'slices.h5')
            print('In-memory dataset creation done. You can now use the dataset for your experiments.')


class CircuitInMemFCDataset(CircuitInMemDataset):

    def get(self, idx):
        data = super().get(idx)
        data.edge_index = dense_to_sparse(torch.ones((data.num_nodes, data.num_nodes)))[0]
        return data
        

# Ad-hoc artificial labeling methods
def meet_spec(spec_dict):
    return dict(
        cost=float(spec_dict['cost'] < 0.05),
        gain=float(spec_dict['gain'] > 300),
        ugbw=float(spec_dict['ugbw'] > 10e6),
        ibias=float(spec_dict['ibias'] < 0.2e-3),
        pm=float(spec_dict['pm'] > 60),
        tset=float(spec_dict['tset'] < 60e-9),
        psrr=float(spec_dict['psrr'] > 50),
        cmrr=float(spec_dict['cmrr'] > 50),
        offset_sys=float(spec_dict['offset_sys'] < 1e-3),
    )

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root')
    parser.add_argument('--mode', default='train')
    _args = parser.parse_args()

    dataset = CircuitGraphDataset(_args.root, mode=_args.mode)
    breakpoint()