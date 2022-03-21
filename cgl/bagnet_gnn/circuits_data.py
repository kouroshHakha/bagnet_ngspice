""" Data classes needed for constructing the graph data structure from the netlist information """
from dataclasses import dataclass
from typing import Dict, Any
import networkx as nx
from copy import copy


@dataclass(frozen=True, eq=True)
class Node:
    name: str # unique name in format of {type}_devName_TerminalType: eg. T_M1_D or V_1
    type: str  # [V | T]

# ------------------------------------- VNode vs. Terminal
@dataclass(frozen=True, eq=True)
class Terminal(Node):
    device_class: str # [M | R | C | V | I]
    vnode: str
    params: Dict[str, Any]

@dataclass(frozen=True, eq=True)
class VNode(Node):
    pass

# ------------------------------- DEVICE Classes
@dataclass
class Device:
    name: str
    params: Dict[str, Any]
    terminals: Dict[str, str]

    @property
    def type(self):
        raise NotImplementedError

    def get_terminals(self):
        terminals = []
        for term_name, vnode in self.terminals.items():
            name = f'T_{self.name}_{term_name}'
            term_param = copy(self.params)
            term_param.update(terminal_type=term_name)
            terminals.append(
                Terminal(name=name, type='T', device_class=self.type, vnode=vnode, params=term_param)
            )
        return terminals


class DevTransistor(Device):

    @property
    def type(self):
        return 'M'

class DevCS(Device):

    @property
    def type(self):
        return 'I'

class DevVS(Device):

    @property
    def type(self):
        return 'V'

class DevRes(Device):

    @property
    def type(self):
        return 'R'

class DevCap(Device):

    @property
    def type(self):
        return 'C'

class Netlist:

    _CLASS_LOOKUP = dict(
        M=DevTransistor,
        I=DevCS,
        V=DevVS,
        R=DevRes,
        C=DevCap,
    )

    def __init__(self, netlist_conf: Dict[str, Any]) -> None:
        
        
        graph = nx.Graph()
        graph_terminals = []
        for name, dev_kwargs in netlist_conf.items():
            dev_cls = self._CLASS_LOOKUP[dev_kwargs['type']]
            dev = dev_cls(name=name, params=dev_kwargs['params'], terminals=dev_kwargs['terminals'])
            terminals = dev.get_terminals()

            sub_nodes = []
            for terminal in terminals:
                props = copy(terminal.params)
                props.update(device_class=terminal.device_class)
                graph.add_node(terminal.name, type='T', props=props)

                for prev_node in sub_nodes:
                    graph.add_edge(terminal.name, prev_node)

                sub_nodes.append(terminal.name) 

            graph_terminals += terminals 
        
        for terminal in graph_terminals:
            vnode = VNode(name=f'V_{terminal.vnode}', type='V')
            props = dict(is_gnd=(terminal.vnode == '0'))
            graph.add_node(vnode.name, type='V', props=props)
            graph.add_edge(terminal.name, vnode.name)
        
        self._graph = graph


    @property
    def graph(self):
        return self._graph
