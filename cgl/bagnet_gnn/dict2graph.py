
from pathlib import Path

import torch
from torch_geometric.data import Data

from cgl.utils.trie import Trie
from cgl.utils.params import ParamDict
from cgl.utils.file import read_yaml
from cgl.data.graph_data import modify_content, normalize
from cgl.bagnet_gnn.circuits_data import Netlist


class Dict2Graph:

    def __init__(self, meta_data_path) -> None:
        
        # load pretraining meta data for normalization and graph encoding consistency
        self.meta_data_path = Path(meta_data_path)

        stats_fpath = self.meta_data_path / 'graph_stats.yaml'
        trie_fpath = self.meta_data_path / 'node_type.trie'

        self._stats = read_yaml(stats_fpath)
        self.node_type_trie = Trie.load(trie_fpath)


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
        # data.x_list = feature_list
        data.x = torch.nn.utils.rnn.pad_sequence(feature_list, batch_first=True)
        # data.type_list = type_list
        data.type_tens = torch.nn.utils.rnn.pad_sequence(type_enc_list, batch_first=True)
        # circuit_params = graph_dict['circuit_params']
        # data.circuit_params = torch.tensor(list(circuit_params.values()), dtype=torch.float64)

        data.output_node_mask = output_node_mask.bool()
        data.output_current_plus = output_current_plus.bool()
        data.output_current_minus = output_current_minus.bool()

        return data, node_map


    def param_to_graph(self, params):
        netlist_dict = get_netlist_dict(params)
        netlist = Netlist(netlist_dict)
        graph = netlist.graph

        graph_dict = {
            'nodes': [dict(name=node, type=graph.nodes[node]['type'], 
                           props=graph.nodes[node].get('props', {})) for node in graph],
            'edges': list(graph.edges())
        }

        updated_graph_dict = modify_content([graph_dict])[0]

        torch_graph, _ = self._process_structural_data_from_graph_dict(updated_graph_dict)

        return torch_graph


def get_netlist_dict(params):

    ans = dict(
        mn1=dict(type='M', params=dict(is_nmos=True, w=int(params['mn1'])), terminals=dict(D='net4', G='net2', S='net3', B='vss')),
        mn2=dict(type='M', params=dict(is_nmos=True, w=int(params['mn1'])), terminals=dict(D='net5', G='net1', S='net3', B='vss')),
        mn3=dict(type='M', params=dict(is_nmos=True, w=int(params['mn3'])), terminals=dict(D='net3', G='net7', S='vss', B='vss')),
        mn4=dict(type='M', params=dict(is_nmos=True, w=int(params['mn4'])), terminals=dict(D='net7', G='net7', S='vss', B='vss')),
        mn5=dict(type='M', params=dict(is_nmos=True, w=int(params['mn5'])), terminals=dict(D='net6', G='net7', S='vss', B='vss')),
        mp1=dict(type='M', params=dict(is_nmos=False, w=int(params['mp1'])), terminals=dict(D='net4', G='net4', S='vdd', B='vdd')),
        mp2=dict(type='M', params=dict(is_nmos=False, w=int(params['mp1'])), terminals=dict(D='net5', G='net4', S='vdd', B='vdd')),
        mp3=dict(type='M', params=dict(is_nmos=False, w=int(params['mp3'])), terminals=dict(D='net6', G='net5', S='vdd', B='vdd')),
        ibias=dict(type='I', params=dict(dc=30e-6, ac_mag=0, ac_ph=0), terminals=dict(PLUS='vdd', MINUS='net7')),
        vvdd=dict(type='V', params=dict(dc=1.2, ac_mag=0, ac_ph=0), terminals=dict(PLUS='vdd', MINUS='0')),
        vvss=dict(type='V', params=dict(dc=0, ac_mag=0, ac_ph=0), terminals=dict(PLUS='vss', MINUS='0')),
        vin1=dict(type='V', params=dict(dc=0.6, ac_mag=1, ac_ph=0), terminals=dict(PLUS='net1', MINUS='0')),
        vin2=dict(type='V', params=dict(dc=0.6, ac_mag=1, ac_ph=0), terminals=dict(PLUS='net2', MINUS='0')),
        rz=dict(type='R', params=dict(value=params['rz']), terminals=dict(PLUS='net8', MINUS='net6')),
        cc=dict(type='C', params=dict(value=params['cc']), terminals=dict(PLUS='net5', MINUS='net8')),
        CL=dict(type='C', params=dict(value=10e-12), terminals=dict(PLUS='net6', MINUS='0')),
    )

    return ans



if __name__ == '__main__':

    params = dict(
        mn1=1,
        mn3=1,
        mn4=1,
        mn5=1,
        mp1=1,
        mp3=1,
        rz=500,
        cc=5e-12,
    )
    converter = Dict2Graph('datasets/bagnet_gnn')
    graph = converter.param_to_graph(params)