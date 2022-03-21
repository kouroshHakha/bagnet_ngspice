from os import write
from typing import Tuple
from cgl.utils.encode import one_hot
from cgl.utils.file import write_yaml, read_yaml

class TrieNode:
    def __init__(self, name='') -> None:
        self.name = name
        self.children = {}
        self.children_order = {}
        self.is_leaf = False
        self.branch_factor = 0

    def add(self, trie_node):
        self.children[trie_node.name] = trie_node
        self.children_order[trie_node.name] = self.branch_factor
        self.branch_factor += 1


    def get_leaf_encodings(self):
        
        if self.is_leaf:
            return {}

        enc_dict = {}
        for char in self.children:
            child_enc = self.children[char].get_leaf_encodings()
            char_enc = one_hot(self.children_order[char], self.branch_factor)
            
            if child_enc:
                for sub_key, sub_enc in child_enc.items():
                    key = (char,) + sub_key
                    enc_dict[key] = char_enc + sub_enc
            else:
                enc_dict[(char,)] = char_enc

        return enc_dict



class Trie:
    def __init__(self) -> None:
        self.root = TrieNode(name='root')

    def add(self, hier: Tuple[str, ...]):
        node = self.root
        for char in hier:
            if char not in node.children:
                node.add(TrieNode(name=char))

            node = node.children[char]
        # when loop is finished the ptr to node is a leaf
        node.is_leaf = True                

    def get_leaf_encodings(self):
        return self.root.get_leaf_encodings()

    def save(self, fname):
        # TODO: This method will ruin the order of the Trie when loaded, but it's fine for our application
        content = self.get_leaf_encodings()
        write_yaml(fname, list(content.keys()))

    @classmethod
    def load(cls, fname):
        node_list = read_yaml(fname)
        trie = Trie()
        for node in node_list:
            trie.add(node)
        return trie

if __name__ == '__main__':
    import pprint

    trie = Trie()
    trie.add(('M', 'N', 'D'))
    trie.add(('M', 'N', 'G'))
    trie.add(('M', 'N', 'S'))
    trie.add(('M', 'N', 'B'))
    trie.add(('M', 'P', 'D'))
    trie.add(('M', 'P', 'G'))
    trie.add(('M', 'P', 'S'))
    trie.add(('M', 'P', 'B'))
    trie.add(('R', 'P'))
    trie.add(('R', 'M'))
    trie.add(('C', 'P'))
    trie.add(('C', 'M'))
    trie.add(('V', 'P'))
    trie.add(('V', 'M'))
    trie.add(('I', 'P'))
    trie.add(('I', 'M'))

    enc_dict = trie.root.get_leaf_encodings()
    pprint.pprint(enc_dict)