

class BinNode:
    def __init__(self, v: int, l=None, r=None):
        self.v = v
        self.l: None | BinNode = l
        self.r: None | BinNode = r

    def fill_rand(self, min_size=5, max_size=10):
        import random
        vals = random.sample([i for i in range(100) if i != self.v], random.randint(min_size, max_size))
        for v in vals:
            self.insert(BinNode(v))
        

    def fill_vals_U(self, vals: list):
        queue = [self]
        while len(queue) > 0:
            curr = queue.pop(0)

            if len(vals) > 0:
                curr.l = BinNode(vals.pop(0))
                queue.append(curr.l)
            if len(vals) > 0:
                curr.r = BinNode(vals.pop(0))
                queue.append(curr.r)

    def insert(self, v):
        if self.v is not None:
            if v < self.v:
                if self.l is None:
                    self.l = BinNode(v)
                else:
                    self.l.insert(v)
            elif v > self.v:
                if self.r is None:
                    self.r = BinNode(v)
                else:
                    self.r.insert(v)    

    def invert(self):
        def _invert(node):
            l, r = node.l, node.r 
            node.l, node.r = r, l
            if l is not None:
                _invert(l)
            if r is not None:
                _invert(r)
        _invert(self)

    
    def print_tree(self):
        queue = [self]

        while queue:
            same_level = []

            for _ in range(len(queue)):
                curr = queue.pop(0)
                same_level.append(curr.v)

                if curr.l is not None:
                    queue.append(curr.l)
                if curr.r is not None:
                    queue.append(curr.r)
            
            print(*same_level)




root = BinNode(0)

for v in [3, 4]:
    root.insert(v)

root.print_tree()
