

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


class MinHeap:
    def __init__(self):
        self.heap = []
    
    def insert(self, val):
        self.heap.append(val)
        i = len(self.heap) - 1
        while i > 0 and self.heap[(i - 1) // 2] > self.heap[i]:
            self.heap[i], self.heap[(i - 1) // 2] = self.heap[(i - 1) // 2], self.heap[i]
            i = (i - 1) // 2

    def heapify(self):
        def _heapify(i):
            mn = i
            l = i * 2 + 1
            r = i * 2 + 2
            if l < len(self.heap) and self.heap[mn] > self.heap[l]:
                mn = l
            if r < len(self.heap) and self.heap[mn] > self.heap[r]:
                mn = r
            if mn != i:
                self.heap[i], self.heap[mn] = self.heap[mn], self.heap[i]
                _heapify(mn)
        self.heapify(0)

def heapify(arr):
    def _heapify(i):
        mn = i
        l = i * 2 + 1
        r = i * 2 + 2
        if l < len(arr) and arr[mn] > arr[l]:
            mn = l
        if r < len(arr) and arr[mn] > arr[r]:
            mn = r
        if mn != i:
            arr[i], arr[mn] = arr[mn], arr[i]
            _heapify(mn)
    for _ in range(len(arr)):
        _heapify(0)
    