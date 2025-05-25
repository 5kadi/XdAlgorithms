
#кодирование гофмана (хаффмана)
class Node:
    def __init__(self, l=None, r=None):
        self.l = l
        self.r = r

    def get_children(self) -> tuple:
        return (self.l, self.r)

def count_frequency(string: str) -> list[tuple]:
    freq_dict = {}
    for char in string:
        if char not in freq_dict:
            freq_dict[char] = 1
        else:
            freq_dict[char] += 1
    freq_dict = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
    return freq_dict

def create_node_tree(freq: list) -> Node:
    node_tree = freq
    while len(node_tree) > 1:
        (k1, v1) = node_tree[-1]
        (k2, v2) = node_tree[-2]
        node_tree = node_tree[:-2]
        node_tree.append((Node(k1, k2), v1 + v2))
        node_tree = sorted(node_tree, key=lambda x: x[1], reverse=True)
    return node_tree[0][0]

def create_huffman_dict(node: Node, curr_code: str="", res: list=[]) -> dict:
    if type(node) == str:
        res.append((node, curr_code))
        return
    
    create_huffman_dict(node.l, curr_code + "0", res)
    create_huffman_dict(node.r, curr_code + "1", res)

    return dict(res)

def print_node_tree(node: Node, pos: str="") -> None:
    if type(node) == str:
        print(pos, node)
        return
    
    print_node_tree(node.l, pos + "l")
    print_node_tree(node.r, pos + "r")

    return