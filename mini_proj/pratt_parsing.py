from string import ascii_letters, digits

class Element:
    def __init__(self, v: str):
        self.v = v

    def __str__(self):
        return self.v

class Atom(Element):
    pass

class Operand(Element):
    def __init__(self, v: str, l=None, r=None):
        self.v = v
        self.l = l
        self.r = r


BINDING_POWER = {
    '+': (1.1, 1),
    '-': (1.1, 1),
    '*': (2.1, 2),
    '/': (2.1, 2),
    '^': (3.1, 3),
}

OPERATIONS = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: a / b,
    '^': lambda a, b: a ** b,
}

def convert(expr: str) -> list[Element]:
    arr = []
    for c in expr:
        if c in ascii_letters or c in digits:
            arr.append(Atom(c))
        else:
            arr.append(Operand(c))
    return arr 

def create_tree(arr: list[Element], prev_bp: int = 0):
    left = arr.pop(0)
    if left.v == '(':
        left = create_tree(arr)

    while len(arr) > 1:
        curr = arr.pop(0)
        if curr.v == ')':
            break

        l_bp, r_bp = BINDING_POWER[curr.v]
        if prev_bp > l_bp:
            arr.insert(0, curr)
            break
        else:
            right = create_tree(arr, r_bp)
            curr.l, curr.r = left, right
            left = curr

    return left

def compute_tree(node: Operand):
    if isinstance(node, Atom):
        return int(node.v)
    
    left_v = compute_tree(node.l)
    right_v = compute_tree(node.r)

    return OPERATIONS[node.v](left_v, right_v)

def print_tree(root: Operand):
    queue = [root]
    while queue:
        curr_lvl = []
        for _ in range(len(queue)):
            node = queue.pop(0)
            curr_lvl.append(node.v)

            if isinstance(node, Operand):
                if node.l is not None:
                    queue.append(node.l)
                if node.r is not None:
                    queue.append(node.r)
        print(*curr_lvl)


expr = "(1+2)^3/4"
expr_conv = convert(expr)

root = create_tree(expr_conv)
print_tree(root)

res = compute_tree(root)
print(res)
