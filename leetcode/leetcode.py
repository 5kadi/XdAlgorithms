from functools import cache
from heapq import *
import itertools
import collections
import bisect
import copy
from math import inf, ceil, floor, log2, gcd, comb, factorial, dist



class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    #ADD
    def __str__(self):
        vals = []
        head = self 
        while head: 
            vals.append(str(head.val))
            head = head.next
        return " ".join(vals)      

    #ADD
    @staticmethod
    def create(vals: list[int]) -> 'ListNode':
        head = res = ListNode()
        for i in range(len(vals)):
            head.next = ListNode(vals[i])
            head = head.next
        return res.next


class Interval:
    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end

class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    #ADD
    @staticmethod
    def create_inorder(vals: list[int]) -> 'TreeNode':
        vals = vals.copy()
        root = TreeNode(vals.pop(0))
        q = [root]
        while vals:
            curr = q.pop(0)
            if vals:
                v = vals.pop(0)
                if v: 
                    curr.left = TreeNode(v)
                    q.append(curr.left)
            if vals:
                v = vals.pop(0)
                if v: 
                    curr.right = TreeNode(v)
                    q.append(curr.right)
        return root
    
    #ADD
    def print_tree(self):
        q = [self]
        while q:
            vals = []
            for _ in range(len(q)):
                node = q.pop(0)
                vals.append(node.val if node else "#")
                if node:
                    q.append(node.left)
                    q.append(node.right)
            print(*vals)

class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

class GraphNode:
    def __init__(self, val: int = 0, neighbors: list['GraphNode'] = []):
        self.val = val
        self.neighbors = neighbors 

    #ADD
    @staticmethod
    def create_adj(adj_list: list[list[int]]) -> 'GraphNode':
        nodes = [None] * (len(adj_list) + 1) #1-indexed
        for i in range(len(adj_list)):
            nodes[i + 1] = GraphNode(i + 1, adj_list[i])

        for n in nodes[1:]:
            neighbors = n.neighbors
            for i in range(len(neighbors)):
                neighbors[i] = nodes[neighbors[i]]
        return nodes[1]


#https://leetcode.com/problems/valid-parentheses/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def isValid(s):
    stack = []
    mapping = {"(":")", "{":"}", "[":"]"}

    for c in s:
        if c in mapping:
            stack.append(c)
        else:
            if not stack:
                return False

            prev = stack.pop()
            if mapping[prev] != c:
                return False  

    return not stack

#https://leetcode.com/problems/two-sum/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def twoSum(nums, target):
    memo = {}
    for i in range(len(nums)):
        if nums[i] in memo:
            return [i, memo[nums[i]]]
        
        memo[target - nums[i]] = i

#https://leetcode.com/problems/merge-two-sorted-lists/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def mergeTwoLists(list1, list2):
    dum = ListNode()
    res = dum
    while list1 and list2:
        if list1.val > list2.val:
            res.next = list2
            list2 = list2.next
        else:
            res.next = list1
            list1 = list1.next
        res = res.next

    if not list1:
        res.next = list2
    else:
        res.next = list1

    return dum.next

#idk lmao
def airplaneProblem(flights: list[Interval]) -> int:
    schedule = []
    for interval in flights:
        schedule.append([interval.start, 1])
        schedule.append([interval.end, -1])
    
    schedule = sorted(schedule, key=lambda x: x[0])
    current, maximum = 0, 0
    for _, status in schedule:
        current += status
        maximum = max(maximum, current)
    return maximum

#https://leetcode.com/problems/best-time-to-buy-and-sell-stock/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def maxProfit(prices: list):
    mn = prices[0]
    profit = 0

    for i in range(1, len(prices)):
        if prices[i] < mn:
            mn = prices[i]
        if prices[i] - mn > profit:
            profit = prices[i] - mn
    return profit

def maxProfit(prices: list[int]) -> int:
    res = 0
    l = 0
    for r in range(1, len(prices)):
        if prices[r] < prices[l]: l = r
        else: res = max(res, prices[r] - prices[l])
    return res

#https://leetcode.com/problems/valid-palindrome/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def isPalindrome(s: str):
    import re
    s = re.sub(r'[^a-zA-Z0-9]+', '', s.lower())

    l = 0
    r = len(s) - 1
    while l < r:
        if s[l] == s[r]:
            l += 1
            r -= 1
        else:
            return False
    return True

#https://leetcode.com/problems/invert-binary-tree/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def invertTree(root: TreeNode | None):
    def _invert(root):
        l, r = root.left, root.right
        root.left, root.right = r, l
        if root.left:
            _invert(root.left)
        if root.right:
            _invert(root.right)
    
    if root:
        _invert(root)
    return root

#https://leetcode.com/problems/valid-anagram/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def isAnagram(s: str, t: str): 
    if len(s) != len(t):
        return False
    
    s_u = set(s)
    for c in s_u:
        if s.count(c) != t.count(c):
            return False
    return True

#https://leetcode.com/problems/linked-list-cycle/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def hasCycle(head: ListNode):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

#https://leetcode.com/problems/maximum-depth-of-binary-tree/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def maxDepth(root: TreeNode):
    if root == None:
        return 0
    
    l = 1 + maxDepth(root.left)
    r = 1 + maxDepth(root.right)
    return max(l, r)

#https://leetcode.com/problems/add-two-numbers/
def addTwoNumbers(l1: ListNode, l2: ListNode):
    def _create(l1, l2, summ=0):
        if not l1 and not l2:
            if summ > 0:
                return ListNode(summ, None)
            return None

        if l1:
            summ += l1.val
            l1 = l1.next
        if l2:
            summ += l2.val
            l2 = l2.next

        parent = ListNode(summ % 10, _create(l1, l2, summ // 10))
        return parent
    return _create(l1, l2)

def addTwoNumbers(l1: ListNode, l2: ListNode) -> ListNode:
    carry = 0
    head = dum = ListNode(0)
    while l1 or l2 or carry:
        sm = carry 
        if l1: 
            sm += l1.val
            l1 = l1.next
        if l2:
            sm += l2.val
            l2 = l2.next
        head.next = ListNode(sm % 10)
        head = head.next
        carry = sm // 10
    return dum.next

#https://leetcode.com/problems/single-number/description/
#⭐
def singleNumber(nums: list):
    checked = set()
    for num in nums:
        if num not in checked:
            checked.add(num)
        else:
            checked.remove(num)
    return checked.pop()

def singleNumber(nums: list):
    res = 0
    for num in nums:
        res ^= num #XOR
    return res 


#https://leetcode.com/problems/reverse-linked-list/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
#⭐
def reverseList(head: ListNode):
    prev = None #or next, depends on how you look at this😁
    while head: 
        temp = head.next
        head.next = prev
        prev = head 
        head = temp
    return prev

#https://leetcode.com/problems/majority-element/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def majorityElement(nums: list):
    count = 0
    res = None
    for num in nums:
        if count == 0:
            res = num
        count += 1 if num == res else -1
    return res

#https://leetcode.com/problems/missing-number/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def missingNumber(nums):
    summ = ((len(nums) + 1) / 2.0) * len(nums)
    for num in nums:
        summ -= num
    return int(summ)

def missingNumber(nums: list[int]) -> int:
    r = len(nums)
    for n in range(len(nums)):
        r = r ^ n ^ nums[n] #numbers in both places will cancel out
    return r

#https://leetcode.com/problems/reverse-string/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def reverseString(self, s):
    s_len = len(s)
    l, r, = 0, s_len - 1
    while l < r:
        s[l], s[r] = s[r], s[l]
        l += 1
        r -= 1
    return s

#https://leetcode.com/problems/diameter-of-binary-tree/?source=submission-noac
def diameterOfBinaryTree(root: TreeNode) -> int:
    mx = 0
    def _dfs(node):
        nonlocal mx
        if not node: return 0

        lh, rh = _dfs(node.left), _dfs(node.right)
        mx = max(mx, lh + rh)
        return max(lh, rh) + 1
    _dfs(root)
    return mx

#https://leetcode.com/problems/middle-of-the-linked-list/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def middleNode(head: ListNode) -> ListNode:
    count, mid = 0, 0
    node = head
    while head != None:
        count += 1 #if there is only one element, still counts
        curr_mid = count // 2
        if curr_mid > mid:
            mid = curr_mid
            node = node.next
        head = head.next
    return node

def middleNode(head: ListNode) -> ListNode:
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow

#https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def sortedArrayToBST(nums: list[int]) -> TreeNode:
    def _create(low, high):
        if low > high:
            return None

        mid = (low + high) // 2
        leaf = TreeNode(nums[mid])
        leaf.left = _create(low, mid - 1)
        leaf.right = _create(mid + 1, high)
        return leaf
    return _create(0, len(nums) - 1)

#https://leetcode.com/problems/product-of-array-except-self/description/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def productExceptSelf(nums: list[int]) -> list[int]:
    res = [1] * len(nums)
    
    l = 1
    for i in range(len(nums)):
        res[i] *= l
        l *= nums[i]

    r = 1
    for i in range(len(nums) - 1, -1, -1):
        res[i] *= r
        r *= nums[i]
    
    return res

#https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def maxProfit(prices: list[int]) -> int:
    total = 0
    for i in range(1, len(prices)):
        if prices[i] - prices[i - 1] > 0:
            total += prices[i] - prices[i - 1]
    return total

#https://leetcode.com/problems/house-robber/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def rob(nums: list[int]) -> int: 
    if len(nums) == 1:
        return nums[0]

    max_profits = [1] * len(nums)
    max_profits[0] = nums[0]
    max_profits[1] = max(nums[0], nums[1])

    for i in range(2, len(nums)):
        max_profits[i] = max(max_profits[i - 1], max_profits[i - 2] + nums[i])
    
    return max_profits[-1]

def rob(nums: list[int]) -> int:
    prev_rob = max_rob = 0

    for cur_val in nums:
        temp = max(max_rob, prev_rob + cur_val)
        prev_rob = max_rob
        max_rob = temp
    
    return max_rob

def rob(nums: list[int]) -> int: 
    d = [0] * (len(nums) + 3)
    res = 0
    for i in range(len(nums)):
        d[i + 3] = nums[i] + max(d[i], d[i + 1])
        res = max(res, d[i + 3])
    return res

#https://leetcode.com/problems/number-of-1-bits/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def hammingWeight(n: int) -> int:
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count

#https://leetcode.com/problems/validate-binary-search-tree/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def isValidBST(root: TreeNode) -> bool:
    def _is_valid(root, mn, mx):
        if not root:
            return True
        
        if not (mn < root.val < mx):
            return False

        return _is_valid(root.left, mn, root.val) and _is_valid(root.right, root.val, mx)
    return _is_valid(root, float("-inf"), float("inf")) 

#https://leetcode.com/problems/min-stack/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
class MinStack:
    def __init__(self):
        self.stack = [] #could do with two stacks

    def push(self, val: int) -> None:
        mn = self.getMin()
        if mn == None or val <= mn:
            mn = val
        self.stack.append((val, mn))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        if not self.stack:
            return None
        return self.stack[-1][0]

    def getMin(self) -> int:
        if not self.stack:
            return None
        return self.stack[-1][1]

#https://leetcode.com/problems/kth-smallest-element-in-a-bst/   
def kthSmallest(root: TreeNode, k: int) -> int:
    height = 0
    res = None
    def _find(root):
        if root == None:
            return
        nonlocal height, res
        
        _find(root.left)
        height += 1
        if height == k:
            res = root.val
        _find(root.right)
    _find(root)
    return res

#https://leetcode.com/problems/merge-k-sorted-lists/
def mergeKLists(lists: list[ListNode]) -> ListNode:
    from heapq import heappush, heappop

    llen = len(lists)
    heap = []

    for i in range(llen):
        head = lists[i]
        while head:
            heappush(heap, head.val)
            head = head.next
    if not heap:
        return None

    head = dum = ListNode(heappop(heap))
    while heap:
        head.next = ListNode(heappop(heap))
        head = head.next
    
    return dum

#https://leetcode.com/problems/merge-intervals/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def merge(intervals: list[list[int]]) -> list[list[int]]:
    intervals.sort(key=lambda x: x[0])
    res = [intervals[0]]

    for i in range(1, len(intervals)):
        if intervals[i][0] <= res[-1][1]: #start2 <= end1
            if intervals[i][1] <= res[-1][1]: #end2 <= end1
                pass #don't append to res
            else:
                res[-1] = [res[-1][0], intervals[i][1]]
        else:
            res.append(intervals[i])

    return res

def merge(intervals: list[list[int]]) -> list[list[int]]:
    t = sorted(intervals)
    curr = t.pop(0)
    res = []
    for s, e in t:
        if s <= curr[1]: curr[1] = max(curr[1], e)
        else:
            res.append(curr)
            curr = [s, e]
    res.append(curr)
    return res   

#https://leetcode.com/problems/set-matrix-zeroes/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def setZeroes(matrix: list[list[int]]) -> None:
    """
    Do not return anything, modify matrix in-place instead.
    """
    if not matrix or not matrix[0]:
        return
    m, n = len(matrix), len(matrix[0])
    zero_rows, zero_cols = set(), set()
    # Pass 1: Traverse through the matrix to identify the rows and columns
    # containing zeros and store their indexes in the appropriate hash sets.
    for r in range(m):
        for c in range(n):
            if matrix[r][c] == 0:
                zero_rows.add(r)
                zero_cols.add(c)
    # Pass 2: Set any cell in the matrix to zero if its row index is in 'zero_rows'
    # or its column index is in 'zero_cols’.
    for r in range(m):
        for c in range(n):
            if r in zero_rows or c in zero_cols:
                matrix[r][c] = 0

def setZeroes(matrix: list[list[int]]) -> None:
    w = len(matrix[0])
    h = len(matrix)

    y_0 = False
    x_0 = False
    
    for y in range(h):
        if matrix[y][0] == 0:
            y_0 = True
    for x in range(w):
            if matrix[0][x] == 0:
                x_0 = True

    for y in range(1, h):
        for x in range(1, w):
            if matrix[y][x] == 0:
                matrix[0][x] = 0
                matrix[y][0] = 0
    
    #fill x
    for y in range(1, h):
        if matrix[y][0] == 0:
            for x in range(1, w):
                matrix[y][x] = 0
    #fill y
    for x in range(1, w):
        if matrix[0][x] == 0:
            for y in range(1, h):
                matrix[y][x] = 0
    
    if y_0:
        for y in range(h):
            matrix[y][0] = 0
    if x_0:
        for x in range(w):
            matrix[0][x] = 0

    return matrix

def setZeroes(matrix: list[list[int]]) -> None:
    w, h = len(matrix[0]), len(matrix)
    xz, yz = set(), set()

    for y in range(h):
        for x in range(w):
            if matrix[y][x] == 0:
                yz.add(y)
                xz.add(x)

    for y in range(h):
        for x in range(w):
            if (x in xz) or (y in yz): matrix[y][x] = 0

#https://leetcode.com/problems/spiral-matrix/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
#⭐(dx, dy = -dy, dx)
def spiralOrder(matrix: list[list[int]]) -> list[int]:
    w = len(matrix[0])
    h = len(matrix)
    res = []
    x_bias = 0
    y_bias = 0

    while len(res) < w * h:
        for i in range(x_bias, w - x_bias):
            res += [matrix[y_bias][i]]
        if len(res) == w*h:
            return res

        y_bias += 1
        for j in range(y_bias, h - y_bias + 1):
            res += [matrix[j][i]] 
        if len(res) == w*h:
            return res

        x_bias += 1
        for i in range(w - x_bias - 1, x_bias - 2, -1):
            res += [matrix[j][i]]
        if len(res) == w*h:
            return res

        for j in range(h - y_bias - 1, y_bias - 1, -1):
            res += [matrix[j][i]]

    return res

def spiralOrder(matrix: list[list[int]]) -> list[int]:
    dx = 1
    dy = 0
    x = 0
    y = 0

    rows, cols = len(matrix), len(matrix[0])
    res = []

    for _ in range (rows * cols):
        res.append(matrix[y][x])
        matrix[y][x] = "." # stop indicator

        if ((0 > x + dx) or (x + dx >= cols)) or ((0 > y + dy) or (y + dy >= rows)) or matrix[y+dy][x+dx] == ".":
            dx, dy = -dy, dx # flip direction

        x += dx
        y += dy

    return res

def spiralOrder(matrix: list[list[int]]) -> list[int]:
    w, h = len(matrix[0]), len(matrix)
    res = []

    chk = set()
    dx, dy = 1, 0
    x, y = -1, 0
    while len(res) < w * h:
        nx, ny = x + dx, y + dy
        if not (0 <= nx < w) or not (0 <= ny < h) or ((nx, ny) in chk):
            dx, dy = -dy, dx
            continue
        res.append(matrix[ny][nx]) 
        chk.add((nx, ny))
        x, y = nx, ny
    return res

#https://leetcode.com/problems/longest-consecutive-sequence/submissions/1708771904/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def longestConsecutive(nums: list[int]) -> int:
    nums = set(nums)
    mx = 0
    for num in nums:
        if num - 1 in nums:
            continue
        curr = 0
        while num in nums:
            num += 1
            curr += 1
        mx = max(curr, mx)
    return mx

def longestConsecutive(nums: list[int]) -> int:
    nums = set(nums)
    best = 0

    for x in nums:
        if x - 1 not in nums:
            y = x + 1
            while y in nums:
                y += 1
            best = max(best, y - x)

    return best

def longestConsecutive(nums: list[int]) -> int:
    if not nums: return 0
    nums = set(nums)
    d = {}
    mx = 0
    for n in nums:
        if n in d: continue
        orig = n
        d[orig] = 1
        while n - 1 in nums:
            n = n - 1
            if n in d:
                d[orig] += d[n]
                break
            else:
                d[orig] += 1
                d[n] = 1  
        mx = max(mx, d[orig])     
    return mx

#https://leetcode.com/problems/3sum/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def threeSum(nums: list[int]) -> list[list[int]]:
    nums = sorted(nums)
    res = []

    for i in range(len(nums)): 
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        j = i + 1
        k = len(nums) - 1

        while j < k:
            summ = nums[i] + nums[j] + nums[k]
            if summ < 0:
                j += 1
            elif summ > 0:
                k -= 1
            else:
                res.append([nums[i], nums[j], nums[k]])
                j += 1

                while nums[j] == nums[j - 1] and j < k:
                    j += 1
    return res

#barely optimized brutforce solution, but it's honest work
def threeSum(nums: list[int]) -> list[list[int]]:
    nums = sorted(nums)
    res = set()
    for l in range(len(nums) - 2):
        m = l + 1
        r = len(nums) - 1
        while m < r:
            sm = nums[l] + nums[m] + nums[r]
            if sm == 0: 
                res.add((nums[l], nums[m], nums[r]))
                m += 1
            elif sm < 0: m += 1
            elif sm > 0: r -= 1
    return [list(x) for x in res]

#https://leetcode.com/problems/climbing-stairs/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def climbStairs(n: int) -> int:
    if n <= 2:
        return n

    n_2 = 1 # n - 1
    n_1 = 2 # n - 2
    for i in range(3, n + 1):
        n = n_1 + n_2
        n_2 = n_1
        n_1 = n
    return n_1

def climbStairs(n: int) -> int:
    if n <= 2: return n

    n2, n1 = 1, 2
    for _ in range(n - 2):
        t = n2 + n1
        n2, n1 = n1, t
    return n1

#https://leetcode.com/problems/symmetric-tree/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def isSymmetric(root: TreeNode) -> bool:
    def _solve(l, r):
        if l == None or r == None:
            return l == r
        return l.val == r.val and _solve(l.left, r.right) and _solve(r.left, l.right)
    return _solve(root.left, root.right)

#https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def zigzagLevelOrder(root: TreeNode) -> list[list[int]]:
    if root == None:
        return []

    res = []
    queue = [root]
    depth = 1
    while queue:
        temp = []
        for _ in range(len(queue)):
            node = queue.pop(0)
            temp.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        if temp:
            if not depth % 2: # = 0, not 0 = True
                res.append(temp[::-1])
            else:
                res.append(temp)
        depth += 1  
    return res

#https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def buildTree(preorder: list[int], inorder: list[int]) -> TreeNode:
    inorder_idx = {val:idx for idx, val in enumerate(inorder)}
    def _build(l, r):
        if l > r:
            return None

        head = TreeNode(preorder.pop(0))
        idx = inorder_idx[head.val]
        head.left = _build(l, idx - 1)
        head.right = _build(idx + 1, r)

        return head   
    return _build(0, len(inorder) - 1)

#https://leetcode.com/problems/container-with-most-water/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def maxArea(height: list[int]) -> int:
    i = 0
    j = len(height) - 1
    max_v = 0
    while i < j:
        l = height[i]
        r = height[j]
        v = min(l, r) * (j - i)
        if v >= max_v:
            max_v = v
        
        if l >= r: #if we hold on smallest capacity, volume will always decrease (maximizing height might lead to higher volume)
            j -= 1
        else:
            i += 1
    return max_v

def maxArea(height: list[int]) -> int:
    mx = 0
    l, r = 0, len(height) - 1
    while l < r:
        h = min(height[l], height[r])
        mx = max(mx, (r - l)*h)
        if height[l] <= height[r]: l += 1
        elif height[r] < height[l]: r -= 1
    return mx

#https://leetcode.com/problems/flatten-binary-tree-to-linked-list/solutions/6673986/master-in-place-tree-flattening-unlock-the-hidden-trick-to-linked-list-conversion/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def flatten(root: TreeNode) -> None:
    arr = []
    def _create(root):
        if root == None:
            return
        
        nonlocal arr
        arr.append(root)
        _create(root.left) 
        _create(root.right)
    
    _create(root)
    if arr:
        arr.pop(0)

    while arr:
        root.right = arr.pop(0)
        root.left = None
        root = root.right

def flatten(root):
    current = root

    while current:
        if current.left:
            predecessor = current.left
            while predecessor.right:
                predecessor = predecessor.right
            predecessor.right = current.right
            current.right = current.left
            current.left = None
        current = current.right

#https://leetcode.com/problems/group-anagrams/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def groupAnagrams(strs: list[str]) -> list[list[str]]:
    from collections import defaultdict
    groups = defaultdict(list)
    for s in strs:
        s_t = tuple(sorted(s))
        groups[s_t].append(s)
        
    return list(groups.values())

def groupAnagrams(strs: list[str]) -> list[list[str]]:
    groups = defaultdict(list)
    for s in strs:
        cnt = [0] * 26
        for c in s:
            cnt[ord(c) - ord('a')] += 1
        groups[tuple(cnt)].append(s)
    return list(groups.values())

#https://leetcode.com/problems/implement-trie-prefix-tree/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
class Trie:
    def __init__(self):
        self.root = {}

    def insert(self, word: str) -> None:
        curr = self.root
        for c in word:
            if c not in curr:
                curr[c] = {}
            curr = curr[c]
        curr["*"] = '' #end of the word 

    def search(self, word: str) -> bool:
        curr = self.root
        for c in word:
            if c not in curr:
                return False #if no such prefix
            curr = curr[c]
        return "*" in curr #if not end of word

    def startsWith(self, prefix: str) -> bool:
        curr = self.root
        for c in prefix:
            if c not in curr:
                return False
            curr = curr[c]
        return True

#https://leetcode.com/problems/kth-largest-element-in-an-array/solutions/6750670/video-4-solutions-with-sorting-heap-counting-sort-and-quick-select/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def findKthLargest(nums: list[int], k: int) -> int:
    mn = min(nums)
    mx = max(nums)

    cnt = [0] * (mx - mn + 1) #[1, 3] => [0, 0, 0]
    for n in nums:
        cnt[n - mn] += 1

    for i in range(len(cnt) - 1, -1, -1):
        k -= cnt[i]
        if k <= 0:
            return mn + i #[1, 3], [1, 0, 1] => 1 + 2 = 3
        
def findKthLargest(nums: list[int], k: int) -> int:
    import heapq
    min_heap = []

    for num in nums:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap) #it works, because this line removes the smallest element

    return min_heap[0]

#https://leetcode.com/problems/longest-palindromic-substring/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def longestPalindrome(s: str) -> str:
    longest = ""
    def _spread(l, r):
        while l >= 0 and r < len(s) and s[l] == s[r] :
            l -= 1
            r += 1
        return s[l+1:r]
    for i in range(len(s)):
        single = _spread(i, i)
        double = _spread(i, i + 1)
        if len(double) > len(longest):
            longest = double
        if len(single) > len(longest):
            longest = single
    return longest

def longestPalindrome(s: str) -> str:
    res = ""
    def _fill(l, r):
        nonlocal res
        while l >= 0 and r < len(s):
            if s[l] != s[r]: break
            l -= 1
            r += 1
        res = max(res, s[l + 1:r], key=len)

    for i in range(len(s)): 
        _fill(i, i)
        _fill(i, i + 1)
    return res

def longestPalindrome(s: str) -> str:
    res_l = res_r = 0
    d = [[False] * len(s) for _ in range(len(s))] #dp[i][j] = True if s[i:j + 1] is palindrome

    for l in range(len(s) - 1, -1, -1):
        for r in range(l, len(s)):
            if s[l] == s[r] and (r - l + 1 <= 3 or d[l + 1][r - 1]): #d[l + 1][r - 1] = True if inside string is palindrome
                d[l][r] = True
                if (r - l + 1) >= (res_r - res_l + 1): res_l, res_r = l, r
    return s[res_l:res_r + 1]

#https://leetcode.com/problems/palindromic-substrings/
def countSubstrings(s: str) -> int:
    res = 0
    d = [[False] * len(s) for _ in range(len(s))]

    for l in range(len(s) - 1, -1, -1):
        for r in range(l, len(s)):
            if s[l] == s[r] and (r - l + 1 <= 3 or d[l + 1][r - 1]):
                d[l][r] = True
                res += 1
    return res

#https://leetcode.com/problems/longest-substring-without-repeating-characters/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def lengthOfLongestSubstring(s: str) -> int:
    memory = {}
    l = 0
    longest = 0
    for r in range(len(s)):
        if s[r] in memory and memory[s[r]] >= l: #notice that >=
            l = memory[s[r]] + 1
        memory[s[r]] = r
        longest = max(longest, r - l + 1)
    return longest

def lengthOfLongestSubstring(s: str) -> int:
    free_chars = defaultdict(lambda: True)
    l, m = 0, 0
    for r in range(len(s)):
        while not free_chars[s[r]]:
            free_chars[s[l]] = True
            l += 1
        free_chars[s[r]] = False
        m = max(r - l + 1, m)
    return m

#https://leetcode.com/problems/maximal-square/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def maximalSquare(matrix: list[list[str]]) -> int:
    if matrix is None or len(matrix) < 1:
        return 0
    
    h = len(matrix)
    w = len(matrix[0])
    
    dp = [[0]*(w + 1) for _ in range(h + 1)]
    max_side = 0 
    
    for y in range(h):
        for x in range(w):
            if matrix[y][x] == '1':
                dp[y + 1][x + 1] = min(dp[y][x], dp[y + 1][x], dp[y][x + 1]) + 1 # Be careful of the indexing since dp grid has additional row and column
                max_side = max(max_side, dp[y + 1][x + 1])
            
    return max_side * max_side

#https://leetcode.com/problems/maximal-square/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def maximalSquare(matrix: list[list[str]]) -> int:
    if matrix is None or len(matrix) < 1:
        return 0
    
    h = len(matrix)
    w = len(matrix[0])
    max_side = 0
    
    for y in range(h):
        for x in range(w): 
            matrix[y][x] = 0 if matrix[y][x] == "0" else 1
            if matrix[y][x] == 1 and max_side == 0:
                max_side = 1
    
    for y in range(1, h):
        for x in range(1, w):  
            if matrix[y][x] == 1:
                matrix[y][x] = min(matrix[y - 1][x], matrix[y][x - 1], matrix[y - 1][x - 1]) + 1
                max_side = max(max_side, matrix[y][x])
            
    return max_side * max_side

def maximalSquare(matrix):
    if not matrix or not matrix[0]:
        return 0
    rows, cols = len(matrix), len(matrix[0])
    dp = [[0] * cols for _ in range(rows)]
    max_side = 0
    
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == '1':
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                max_side = max(max_side, dp[i][j])
    
    return max_side * max_side

#https://leetcode.com/problems/maximum-product-subarray/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def maxProduct(nums: list[int]) -> int:
    res = max(nums)
    mx = mn = 1
    for n in nums:
        mx_product = mx * n
        mn_product = mn * n
        mx = max(mx_product, mn_product, n)
        mn = min(mx_product, mn_product, n)

        res = max(res, mx)
    return res

#https://leetcode.com/problems/minimum-window-substring/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def minWindow(s: str, t: str) -> str:
    from collections import defaultdict
    if len(s) < len(t):
        return ""

    count = defaultdict(int)
    for c in t:
        count[c] += 1
    
    mnw = (0, len(s))
    rem = len(t)
    i = 0

    for j, c in enumerate(s):
        if count[c] > 0: #only t char count can be greater than 0
            rem -= 1
        count[c] -= 1

        if rem == 0:
            while True:
                if count[s[i]] == 0: #only t char count can be 0 after i increment
                    count[s[i]] += 1
                    rem += 1
                    break
                count[s[i]] += 1
                i += 1
            if j - i <= mnw[1] - mnw[0]:
                mnw = (i, j)
            i += 1
    return "" if mnw[1] > len(s) - 1 else s[mnw[0]:mnw[1] + 1]

def minWindow(s: str, t: str) -> str:
    if len(t) > len(s): return ""
    elif len(t) == 1: return t if t in s else ""
    
    mn = "*" * (len(s) + 1)
    tcnt = collections.Counter(t)
    cnt = defaultdict(int)
    rem = len(tcnt) #NOTE: rem = len(set(t))

    l = 0
    for r in range(len(s)):
        c = s[r]
        cnt[c] += 1
        if c in tcnt and cnt[c] == tcnt[c]: rem -= 1

        while rem == 0:
            if r - l + 1 < len(mn): mn = s[l:r+1]
            cnt[s[l]] -= 1
            if s[l] in tcnt and cnt[s[l]] < tcnt[s[l]]: rem += 1
            l += 1
    return mn if mn[0] != "*" else ""

#https://leetcode.com/problems/number-of-islands/solutions/6744132/video-check-4-directions-bonus-solutions/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def numIslands(grid: list[list[str]]) -> int:
    w = len(grid[0])
    h = len(grid)
    def _bfs(x, y):
        q = collections.deque()
        q.append((x, y))
        grid[y][x] = "0"
        while q:
            x, y = q.popleft()
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < h) or not (0 <= nx < w) or grid[ny][nx] == "0": continue 
                q.append((nx, ny))
                grid[ny][nx] = "0"

    k = 0  
    for y in range(h):
        for x in range(w):
            if grid[y][x] == "1":
                k += 1
                _bfs(x, y)         
    return k

def numIslands(grid: list[list[str]]) -> int:
    w = len(grid[0])
    h = len(grid)
    def _fill(y, x):
        nonlocal w, h
        if not 0 <= y < h or not 0 <= x < w or grid[y][x] == "0":
            return
        
        grid[y][x] = "0"
        _fill(y + 1, x) 
        _fill(y - 1, x)
        _fill(y, x + 1)
        _fill(y, x - 1)
    
    islands = 0
    for y in range(h):
        for x in range(w):
            if grid[y][x] == "1":
                islands += 1
                _fill(y, x)
    return islands

def numIslands(grid: list[list[str]]) -> int:
    w, h = len(grid[0]), len(grid)
    def _idx(x, y): return y * w + x

    p = list(range(w * h + 1))
    def _find(x):
        if p[x] == x: return x
        else: 
            p[x] = _find(p[x])
            return p[x]
    
    k = 0
    for y in range(h):
        for x in range(w):
            if grid[y][x] == "0": p[_idx(x, y)] = None
            else:
                k += 1
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if not (0 <= ny < h) or not (0 <= nx < w) or grid[ny][nx] == "0": continue 
                    pa, pb = _find(_idx(x, y)), _find(_idx(nx, ny))
                    if pa != pb:
                        p[pb] = pa
                        k -= 1
    return k

#https://leetcode.com/problems/permutations/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def permute(nums: list[int]) -> list[list[int]]:
    res = []
    used = [False] * len(nums)
    
    def backtrack(path,used):
        if len(path) == len(nums):
            res.append(path[:]) #path.pop() changes 
            return 
        for i in range(len(nums)):
            if not used[i]:
                used[i] = True
                path.append(nums[i])
                backtrack(path, used)
                path.pop()
                used[i] = False

    backtrack([], used)
    return res

def permute(nums: list[int]) -> list[list[int]]:
    res = []
    def _permute(curr=[]):
        nonlocal res
        if len(curr) == len(nums):
            res.append(curr)
            return

        for n in nums:
            if n not in curr:
                _permute(curr + [n])
    _permute()
    return res

def permute(nums: list[int]) -> list[list[int]]:
    res = []
    used = [False] * len(nums)
    curr = []
    def _dfs():
        if len(curr) == len(nums):
            res.append(curr.copy())
            return
        
        for i in range(len(nums)):
            if not used[i]:
                used[i] = True
                curr.append(nums[i])
                _dfs()
                curr.pop()
                used[i] = False
    _dfs()
    return res

def permute(nums: list[int]) -> list[list[int]]:
    res = []
    def _dfs(curr, i):
        if i >= len(curr):
            res.append(curr.copy())
            return 

        for j in range(i, len(nums)):
            curr[i], curr[j] = curr[j], curr[i]
            _dfs(curr, i + 1)
            curr[i], curr[j] = curr[j], curr[i]
    _dfs(nums, 0)
    return res

#https://leetcode.com/problems/remove-nth-node-from-end-of-list/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def removeNthFromEnd(head: ListNode, n: int) -> ListNode:
    window = [] #window[0] is element, that is previous to nth from the end
    head = dum = head
    while head:
        if len(window) >= n + 1: #that's why len(window) == n + 1 at max
            window.pop(0)
        window.append(head)
        head = head.next
    
    if len(window) <= 1:
        return 
    else:
        if len(window) == n: #example: [1, 2], n = 2
            return window[1]
        window.append(None)
        window[0].next = window[-n] #we added None at the end, so it's actually -(n - 1)
    return dum

def removeNthFromEnd(head: ListNode, n: int) -> ListNode:
    res = ListNode(0, head)
    dum = res
    for _ in range(n):
        head = head.next #n + 1'th element from the beginning
    
    while head:
        head = head.next #go to the end
        dum = dum.next #go to the n + 1'th element from the beginning
    #there are n + 1 elements between dum and head
    dum.next = dum.next.next #basically window[0].next = window[-n]
    return res.next

def removeNthFromEnd(head: ListNode, n: int) -> ListNode:
    d = ListNode(0, head)
    l, r = d, head

    for _ in range(n - 1): r = r.next

    while r.next: l, r = l.next, r.next

    l.next = l.next.next
    return d.next

#https://leetcode.com/problems/rotate-image/
def rotate(matrix: list[list[int]]) -> None:
    n = len(matrix)
    #transpose
    for y in range(n - 1): #no need reverse [-1][-1] element
        for x in range(y + 1, n): #no need to reverse [0][0] element in current "square"
            matrix[y][x], matrix[x][y] = matrix[x][y], matrix[y][x]  
    #reverse rows
    for y in range(n):
        for x in range(n // 2):
            matrix[y][x], matrix[y][n - 1 - x] = matrix[y][n - 1 - x], matrix[y][x]

#I will post the same code because I can Xd
def rotate(matrix: list[list[int]]) -> None:
    n = len(matrix) #matrix is n * n
    for y in range(n - 1): #transpose
        for x in range(y + 1, n):
            matrix[y][x], matrix[x][y] = matrix[x][y], matrix[y][x]

    for x in range(n // 2): #reverse, x at the tope somehow makes it run faster
        for y in range(n):
            matrix[y][x], matrix[y][(n - x - 1)] = matrix[y][(n - x - 1)], matrix[y][x]
    return matrix    

#https://leetcode.com/problems/search-a-2d-matrix/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def searchMatrix(matrix: list[list[int]], target: int) -> bool:
    h = len(matrix)

    y = 0
    while y < h:
        if matrix[y][0] <= target <= matrix[y][-1]:
            for n in matrix[y]:
                if n == target:
                    return True
            return False
        else:
            y += 1
    return False

def searchMatrix(matrix: list[list[int]], target: int) -> bool:
    w, h = len(matrix[0]), len(matrix)

    l, r = 0, h - 1
    while l <= r:
        m = l + ((r - l) // 2)
        if matrix[m][0] < target: l = m + 1
        elif matrix[m][0] > target: r = m - 1
        else: return True

    row = matrix[l + ((r - l) // 2)]
    l, r = 0, w - 1
    while l <= r:
        m = l + ((r - l) // 2)
        if row[m] < target: l = m + 1
        elif row[m] > target: r = m - 1
        else: return True
    return False

#https://leetcode.com/problems/search-in-rotated-sorted-array/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def search(nums: list[int], target: int) -> int:   
    l = 0
    r = len(nums) - 1
    while l <= r:
        mid = (l + r) // 2
        if nums[mid] == target:
            return mid
        elif nums[l] <= nums[mid]: #mid and l are in the same sorted boundary
            if nums[l] <= target <= nums[mid]:
                r = mid - 1 #search in left boundary
            else:
                l = mid + 1 #search in right boundary
        else: #mid and l are in a different sorted boundaries
            if nums[mid] <= target <= nums[r]:
                l = mid + 1  #search in right boundary
            else:
                r = mid - 1 #search in left boundary
    return -1

#did it myself
def search(nums: list[int], target: int) -> int:   
    first, last = nums[0], nums[-1]
    if target == first: return 0
    elif target == last: return len(nums) - 1

    l, r = 0, len(nums) - 1
    if first < last: # array is sorted
        while l <= r:
            m = l + ((r - l) // 2)
            if nums[m] == target: return m
            elif nums[m] > target: r = m - 1
            elif nums[m] < target: l = m + 1
    else:
        while l <= r:
            m = l + ((r - l) // 2)
            if nums[m] == target: return m
            if nums[m] > target: 
                if target > last: r = m - 1 #first half
                else: #second half
                    if nums[m] >= first: l = m + 1 #not in the second half yet
                    else: r = m - 1  #in the second half
            elif target > nums[m]: 
                if target > last: #first half
                    if nums[m] >= last: l = m + 1 #num is in the first half
                    else: r = m - 1
                else: l = m + 1 #second half
    return -1

#https://leetcode.com/problems/subsets/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
#⭐
def subsets(nums: list[int]) -> list[list[int]]:
    res = [[]]
    used = [False] * len(nums)
    def _create(curr, used):
        nonlocal res
        if all(used):
            return

        for i in range(len(nums)):
            if not used[i]:
                res.append(curr + [nums[i]])
                used[i] = True
                _create(res[-1], used[:])
    _create([], used)
    return res

def subsets(nums: list[int]) -> list[list[int]]:
    res = []
    subset = []

    def create_subset(i):
        if i == len(nums):
            res.append(subset[:])
            return
        
        subset.append(nums[i])
        create_subset(i + 1)

        subset.pop()
        create_subset(i + 1)

    create_subset(0)
    return res

def subsets(nums: list[int]) -> list[list[int]]:
    res = [[]]
    for n in nums:
        res += [sub + [n] for sub in res]
    return res

#https://leetcode.com/problems/subsets-ii/
def subsetsWithDup(nums: list[int]) -> list[list[int]]:
    nums = sorted(nums)
    res = set()
    curr = []
    def _dfs(i):
        if i >= len(nums):
            res.add(tuple(curr.copy()))
            return

        curr.append(nums[i])
        _dfs(i + 1)
        curr.pop()
        _dfs(i + 1)
    _dfs(0)
    return [list(x) for x in res]

def subsetsWithDup(nums: list[int]) -> list[list[int]]:
    nums = sorted(nums)
    res = []
    curr = []
    def _dfs(i):
        res.append(curr.copy())

        for j in range(i, len(nums)):
            if j > i and nums[j] == nums[j - 1]: continue
            curr.append(nums[j])
            _dfs(j + 1)
            curr.pop()
    _dfs(0)
    return res

#https://leetcode.com/problems/top-k-frequent-elements/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def topKFrequent(nums: list[int], k: int) -> list[int]:
    count = {}
    for n in nums:
        count[n] = 1 + count.get(n, 0)
    
    freq = [[] for _ in range(len(nums) + 1)] #bucket sort
    for n, f in count.items():
        freq[f].append(n)
    
    res = []
    for i in range(len(freq) - 1, -1, -1):
        for n in freq[i]:
            res.append(n)
            if len(res) == k:
                return res
            
def topKFrequent(nums: list[int], k: int) -> list[int]:
    from heapq import heappush, heappop
    freq = {}
    for n in nums:
        freq[n] = 1 + freq.get(n, 0)
    
    top_k = []
    for n, f in freq.items():
        heappush(top_k, (-f, n))
    
    res = []
    while len(res) < k:
        res.append(heappop(top_k)[1])
    return res

def topKFrequent(nums: list[int], k: int) -> list[int]:
    cnt = {}
    for n in nums:
        cnt[n] = 1 + cnt.get(n, 0)
    
    v = sorted(cnt.items(), key=lambda x: x[1])
    res = [x[0] for x in v[-k:]]
    return res

#https://leetcode.com/problems/trapping-rain-water/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def trap(height: list[int]) -> int:
    total = 0
    l = 0
    r = len(height) - 1
    l_mx = height[l]
    r_mx = height[r]

    while l < r:
        if l_mx < r_mx:
            l += 1
            l_mx = max(l_mx, height[l])
            total += l_mx - height[l]
        else:
            r -= 1
            r_mx = max(r_mx, height[r])
            total += r_mx - height[r]
    return total

def trap(height: list[int]) -> int:
    l, r = 0, len(height) - 1
    lmx = rmx = 0
    res = 0
    while l < r:
        lmx = max(lmx, height[l])
        rmx = max(rmx, height[r])
        if lmx <= rmx:
            res += lmx - height[l]
            l += 1
        else:
            res += rmx - height[r]
            r -= 1
    return res

#https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def twoSum(numbers: list[int], target: int) -> list[int]:
    i = 0
    j = len(numbers) - 1
    while i < j:
        summ = numbers[i] + numbers[j]
        if summ == target:
            return [i + 1, j + 1]
        elif summ < target:
            i += 1
        else:
            j -= 1

#https://leetcode.com/problems/unique-paths/submissions/1725882067/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def uniquePaths(m: int, n: int) -> int:
    r = [1] * n
    for i in range(m - 1):
        pr = r.copy()
        for j in range(1, n):
            pr[j] = pr[j - 1] + r[j]
        r = pr
    return r[-1]

def uniquePaths(m: int, n: int) -> int:
    from math import factorial
    #m-1+n-1 = m+n-2 total moves for each path
    #let f(r) = (m+n-2)!//(r!*(m+n-2-r))
    #f(m-1) = f(n-1)
    return factorial(m + n - 2) // (factorial(m - 1) * factorial(n - 1))

#https://leetcode.com/problems/valid-sudoku/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def isValidSudoku(board: list[list[str]]) -> bool:
    from collections import defaultdict
    rows = defaultdict(set)
    cols = defaultdict(set)
    boxes = defaultdict(set)

    for y in range(9):
        for x in range(9):
            if board[y][x] == ".":
                continue
            
            if board[y][x] in rows[y] or board[y][x] in cols[x] or board[y][x] in boxes[(y // 3, x // 3)]:
                return False
            
            rows[y].add(board[y][x])
            cols[x].add(board[y][x])
            boxes[(y // 3, x // 3)].add(board[y][x])
    return True

#https://leetcode.com/problems/word-break/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def wordBreak(s: str, wordDict: list[str]) -> bool:
    #  l e e t c o d e
    #  T F F F T F F F T
    memo = [True] + [False] * len(s)
    for i in range(1, len(s) + 1):
        for w in wordDict:
            st = i - len(w)
            if st >= 0 and memo[st] and s[st:i] == w:
                memo[i] = True
                break
    return memo[-1]

def wordBreak(s: str, wordDict: list[str]) -> bool:
    d = [False] * (len(s) + 1) #d[i] = True if s[i:] can be segmented
    d[len(s)] = True
    for i in range(len(s) - 1, -1, -1):
        for w in wordDict:
            end = i + len(w)
            if end <= len(s) and s[i:end] == w:
                d[i] = d[end] #if word can be segmented
            if d[i]: break
    return d[0]

#https://leetcode.com/problems/word-search/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def exist(board: list[list[str]], word: str) -> bool:
    w = len(board[0])
    h = len(board)
    def _find(y, x, i):
        if not 0 <= y < h or not 0 <= x < w or board[y][x] != word[i] or board[y][x] == ".":
            return False
        if i == len(word) - 1:
            return True

        temp = board[y][x]
        board[y][x] = "."
        
        c1 = _find(y - 1, x, i + 1)
        c2 = _find(y + 1, x, i + 1)
        c3 = _find(y, x - 1, i + 1)
        c4 = _find(y, x + 1, i + 1)

        board[y][x] = temp
        return c1 or c2 or c3 or c4
    
    for y in range(h):
        for x in range(w):
            if board[y][x] == word[0]:
                if _find(y, x, 0):
                    return True
    return False

#https://leetcode.com/problems/basic-calculator/submissions/1728196341/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def calculate(s: str) -> int:
    def _calc(i):
        def update(op, v):
            if op == "+":
                stack.append(v)
            if op == "-":
                stack.append(-v)

        num = 0
        stack = []
        sign = "+"

        while i < len(s):
            if s[i].isdigit():
                num = num * 10 + int(s[i])
            elif s[i] in "+-":
                update(sign, num)
                num = 0
                sign = s[i]
            elif s[i] == "(":
                num, j = _calc(i + 1)
                i = j
            elif s[i] == ")":
                update(sign, num)
                return sum(stack), i
            i += 1
        update(sign, num)
        return sum(stack)
    return _calc(0)

def calculate(s: str):
    res = 0
    num = 0
    stack = []
    sign = 1

    for c in s:
        if c.isdigit():
            num = num * 10 + int(c)
        elif c == "+":
            res += sign * num
            num = 0
            sign = 1
        elif c == "-":
            res += sign * num
            num = 0
            sign = -1
        elif c == "(":
            stack.append(res)
            stack.append(sign)

            res = 0
            sign = 1
        elif c == ")":
            res += sign * num
            num = 0

            res *= stack.pop(0)
            res += stack.pop(0)

        res += sign * num
        return res

#https://leetcode.com/problems/coin-change/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def coinChange(coins: list[int], amount: int) -> int:
    d = [0] + [float("inf")] * amount
    for i in range(1, amount + 1):
        for c in coins:
            if i - c >= 0: d[i] = min(d[i], d[i - c] + 1)
    return d[amount] if d[amount] != float("inf") else -1
    
def coinChange(coins: list[int], amount: int) -> int:
    @cache
    def _dfs(rem):
        if rem == 0: return 0
        if rem < 0: return float("inf")

        mnk = float("inf")
        for c in coins:
            mnk = min(_dfs(rem - c), mnk)
        return mnk + 1
    
    res = _dfs(amount)
    if res == float("inf"): return -1
    else: return res
    
#https://leetcode.com/problems/combination-sum/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def combinationSum(candidates: list[int], target: int) -> list[list[int]]:
    res = []
    def _create(target, start, curr=[]):
        nonlocal res
        if target == 0:
            res.append(curr)
            return
        
        for i in range(start, len(candidates)):
            sub = target - candidates[i]
            if sub < 0:
                continue
            _create(sub, i, curr + [candidates[i]])
        return
    _create(target, 0)
    return res

def combinationSum(candidates: list[int], target: int) -> list[list[int]]:
    res = []
    curr = []
    def _dfs(rem, st):
        if rem == 0:
            res.append(curr.copy())
            return
        
        for i in range(st, len(candidates)):
            n = candidates[i]
            if rem - n >= 0: 
                curr.append(n)
                _dfs(rem - n, i)
                curr.pop()
    _dfs(target, 0)
    return res

#https://leetcode.com/problems/combination-sum-ii/
def combinationSum2(candidates: list[int], target: int) -> list[list[int]]:
    candidates = sorted(candidates)
    res = []
    curr = []
    def _dfs(rem, st):
        if rem == 0:
            res.append(curr.copy())
            return
        
        u = defaultdict(lambda: False)
        for i in range(st, len(candidates)):
            n = candidates[i]
            if u[n]: continue
            if rem - n >= 0: 
                curr.append(n)
                _dfs(rem - n, i + 1)
                curr.pop()
                u[n] = True
    _dfs(target, 0)
    return res

#https://leetcode.com/problems/copy-list-with-random-pointer/?source=submission-noac
def copyRandomList(head: Node) -> Node | None:
    if not head:
        return None
    #Idea: 
    #X -> Y -> Z
    #X -> X' -> Y -> Y' -> Z -> Z'
    
    root = head
    while root:
        temp = root.next
        root.next = Node(root.val)
        root.next.next = temp
        root = temp
    
    root = head
    while root:
        if root.random:
            root.next.random = root.random.next
        root = root.next.next
    
    root = dum = head.next
    while root.next:
        root.next = root.next.next
        root = root.next

    return dum

def copyRandomList(head: Node) -> Node | None:
    clone_table = {}
    def _create(head):
        if not head:
            return None
        if head in clone_table:
            return clone_table[head]
        
        clone = Node(head.val)
        clone_table[head] = clone

        clone.next = _create(head.next)
        clone.random = _create(head.random)
        return clone
    return _create(head)

def copyRandomList(head: Node) -> Node | None:
    if not head: return None
    
    copy_map = {}
    node = head
    while node:
        copy_map[node] = Node(node.val)
        node = node.next

    for initial, copy in copy_map.items():
        if initial.next: copy.next = copy_map[initial.next] 
        if initial.random: copy.random = copy_map[initial.random]

    return copy_map[head]

#https://leetcode.com/problems/course-schedule/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-1
def canFinish(numCourses: int, prerequisites: list[list[int]]) -> bool:
    req = defaultdict(list)
    for r, c in prerequisites: req[c].append(r)

    v = set()
    def _dfs(c): #cycle detection
        if c in v: return False
        if not req[c]: return True
        
        v.add(c)
        for r in req[c]:
            if not _dfs(r): return False
        v.remove(c)
        req[c].clear()
        return True
    
    for c in range(numCourses):
        if not _dfs(c): return False
    return True

#BFS
def canFinish(numCourses: int, prerequisites: list[list[int]]) -> bool:
    inp, out = defaultdict(set), defaultdict(set)
    for r, c in prerequisites:
        inp[c].add(r)
        out[r].add(c)

    for x in range(numCourses): #looks like I've reinvented topological sort with this one😭😭😭
        if not inp[x]:
            q = collections.deque([x])
            while q:
                c = q.popleft()
                while out[c]:
                    outc = out[c].pop()
                    inp[outc].remove(c)
                    if not inp[outc]: q.append(outc)
    return all(not inp[c] for c in range(numCourses))

#https://leetcode.com/problems/design-add-and-search-words-data-structure/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
class WordDictionary:
    def __init__(self):
        self.root = {}

    def addWord(self, word: str) -> None:
        root = self.root
        for c in word:
            if c not in root:
                root[c] = {}
            root = root[c]
        root['*'] = True

    def search(self, word: str) -> bool:
        def _search(root, i):
            if i == len(word):
                return root.get('*', False)
            
            if word[i] != '.':
                c = root.get(word[i], False)
                if not c:
                    return False
                else:
                    return _search(c, i + 1)
            else:
                status = False
                for k in root.keys():
                    if k != "*":
                        status = status or _search(root[k], i + 1)
                return status
        return _search(self.root, 0)
    
class WordDictionary:
    def __init__(self):
        self.root = {}

    def addWord(self, word: str) -> None:
        curr = self.root
        for c in word:
            if c not in curr: curr[c] = {}
            curr = curr[c]
        curr["*"] = True

    def search(self, word: str) -> bool:
        n = len(word)
        def _dfs(i, curr):
            if not curr: return False
            if i > n: return False
            if i == n: return "*" in curr
            
            char = word[i]
            if char != ".": return (char in curr) and _dfs(i + 1, curr[char])
            else:
                found = False
                for char in curr: 
                    if char == "*": continue
                    found = found or _dfs(i + 1, curr[char])
                return found
        return _dfs(0, self.root)

#https://leetcode.com/problems/merge-sorted-array/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100 
def merge(nums1: list[int], m: int, nums2: list[int], n: int) -> None:
    mi = m - 1
    ni = n - 1
    i = m + n - 1
    while ni >= 0:
        if mi >= 0 and nums1[mi] > nums2[ni]:
            nums1[i] = nums1[mi]
            mi -= 1
        else:
            nums1[i] = nums2[ni]
            ni -= 1
        i -= 1
    return nums1

#https://leetcode.com/problems/game-of-life/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def gameOfLife(board: list[list[int]]) -> None:
    w = len(board[0])
    h = len(board)
    updated = {}
    def _check(x, y):
        positions = [[-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1]]
        count = 0
        for dx, dy in positions:
            n_x = x + dx
            n_y = y + dy
            if 0 <= n_x < w and 0 <= n_y < h:
                prev = updated.get((n_x, n_y), -1) 
                if prev == 1:
                    count += 1
                else:
                    if board[n_y][n_x] == 1 and prev != 0:
                        count += 1
        return count
    for y in range(h):
        for x in range(w):
            n_c = _check(x, y)
            if board[y][x] == 1:
                if n_c < 2:
                    board[y][x] = 0
                    updated[(x, y)] = 1
                elif n_c > 3:
                    board[y][x] = 0
                    updated[(x, y)] = 1
            else:
                if n_c == 3:
                    board[y][x] = 1
                    updated[(x, y)] = 0
    return board

#https://leetcode.com/problems/find-median-from-data-stream/description/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
class MedianFinder:
    
    def __init__(self):
        self.l = [] #max heap
        self.r = [] #min heap

    def addNum(self, num: int) -> None:
        from heapq import heappop, heappush
        if len(self.l) == len(self.r):
            heappush(self.l, -num)
            lmn = -heappop(self.l)
            heappush(self.r, lmn)
        else:
            heappush(self.r, num)
            rmx = -heappop(self.r)
            heappush(self.l, rmx)

    def findMedian(self) -> float:
        if len(self.l) == len(self.r):
            return (self.r[0] - self.l[0]) / 2
        else:
            return float(self.r[0])

#https://leetcode.com/problems/jump-game/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100       
def canJump(nums: list[int]) -> bool:
    target = len(nums) - 1
    for i in range(len(nums) - 2, -1, -1):
        if nums[i] < target - i:
            continue
        else:
            target = i
    return target == 0
    
def canJump(nums: list[int]) -> bool:
    nlen = len(nums)
    if nlen <= 1:
        return True

    memo = [False]*nlen
    dist = 1
    for i in range(nlen - 2, -1, -1):
        if nums[i] < dist:
            dist += 1
            continue
        else:
            memo[i] = True
            dist = 1
    return memo[0]

def canJump(nums: list[int]) -> bool:
    mxi = nums[0]
    for i in range(1, len(nums)):
        if i <= mxi: mxi = max(mxi, i + nums[i])
        else: return False
    return True

#https://leetcode.com/problems/letter-combinations-of-a-phone-number/submissions/1739711072/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def letterCombinations(digits: str) -> list[str]:
    letters = {
        '2': "abc", '3': "def", '4': "ghi", '5': "jkl", '6': "mno", '7': "pqrs", '8': "tuv", '9': "wxyz"
    }
    res = []
    curr = []

    def _backtrack(i):
        nonlocal res
        if i >= len(digits):
            if len(curr) > 0:
                res.append("".join(curr[:]))
            return
        
        l = letters[digits[i]]
        for c in l:
            curr.append(c)
            _backtrack(i + 1)
            curr.pop()
    _backtrack(0)
    return res

#https://leetcode.com/problems/longest-increasing-subsequence/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def lengthOfLIS(nums: list[int]) -> int:
    res = []
    def _search(arr, n):
        l = 0
        r = len(arr) - 1
        while l <= r:
            mid = (l + r) // 2
            if arr[mid] == n:
                return mid
            elif arr[mid] > n:
                r = mid - 1
            else:
                l = mid + 1
        return l
        
    for n in nums:
        if not res or res[-1] < n:
            res.append(n)
        else:
            i = _search(res, n)
            res[i] = n
    return len(res)

def lengthOfLIS(nums: list[int]) -> int:
    d = [nums[0]]
    for n in nums:
        if n > d[-1]: d.append(n)
        else:
            i = bisect.bisect_left(d, n)
            d[i] = n
    return len(d)

def lengthOfLIS(nums: list[int]) -> int:
    memo = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                memo[i] = max(memo[j] + 1, memo[i])
    return max(memo)

def lengthOfLIS(nums: list[int]) -> int:
    d = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] < nums[j]: d[j] = max(d[i] + 1, d[j])
    return max(d)

#https://leetcode.com/problems/median-of-two-sorted-arrays/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def findMedianSortedArrays(nums1: list[int], nums2: list[int]) -> float:
    m = len(nums1)
    n = len(nums2)
    res = []

    i = j = 0
    while  i < m and j < n:
        if nums1[i] < nums2[j]:
            res.append(nums1[i])
            i += 1
        else:
            res.append(nums2[j])
            j += 1
    res += nums1[i:] + nums2[j:]

    mid = (m + n) // 2
    if not (m + n) % 2:
        return (res[mid] + res[mid - 1]) / 2
    else:
        return float(res[mid])
    
def findMedianSortedArrays(nums1: list[int], nums2: list[int]) -> float:
    m, n = len(nums1), len(nums2)
    i = j = 0
    m1 = m2 = 0
    for c in range((m + n) // 2 + 1):
        m2 = m1 
        if i < m and j < n:
            if nums1[i] > nums2[j]:
                m1 = nums2[j]
                j += 1
            else:
                m1 = nums1[i]
                i += 1
        elif i < m:
            m1 = nums1[i]
            i += 1
        else:
            m1 = nums2[j]
            j += 1
    if (m + n) % 2 == 0 : return (m1 + m2) / 2
    else: return m1

#partition such that all elements to the left are smaller than min element to the right
def findMedianSortedArrays(nums1: list[int], nums2: list[int]) -> float:
    a, b = nums1, nums2
    if len(b) < len(a): a, b = b, a
    total = len(a) + len(b)
    
    l, r = 0, len(a) - 1
    while True:
        ma = l + ((r - l) // 2)
        mb = (total // 2 - (ma + 1)) - 1 #compresses to: total // 2 - 2 - ma

        al = a[ma] if ma >= 0 else float("-inf")
        ar = a[ma + 1] if ma + 1 < len(a) else float("inf")
        bl = b[mb] if mb >= 0 else float("-inf")
        br = b[mb + 1] if mb + 1 < len(b) else float("inf")

        if al <= br and bl <= ar:
            if total % 2 == 0: return (max(al, bl) + min(ar, br)) / 2 #al and bl go to the left
            else: return min(ar, br) #ar and br go to the right
        elif al > br: r = ma - 1
        else: l = ma + 1

#https://leetcode.com/problems/minimum-path-sum/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def minPathSum(grid: list[list[int]]) -> int:
    h = len(grid)
    w = len(grid[0])
    for x in range(1, w):
        grid[0][x] += grid[0][x - 1]
    for y in range(1, h):
        grid[y][0] += grid[y - 1][0]

    for y in range(1, h):
        for x in range(1, w):
            grid[y][x] = grid[y][x] + min(grid[y - 1][x], grid[y][x - 1])
    
    return grid[-1][-1]

#https://leetcode.com/problems/word-search-ii/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def findWords(board: list[list[str]], words: list[str]) -> list[str]:
    w = len(board[0])
    h = len(board)
    trie = {}
    for word in words:
        curr = trie
        for c in word:
            if c not in curr:
                curr[c] = {}
            curr = curr[c]
        curr['*'] = word
    
    res = set()
    def _find(parent, y, x):
        nonlocal res
        if not (0 <= y < h) or not (0 <= x < w):
            return
        
        letter = board[y][x]
        if letter in parent:
            curr = parent[letter]
            if "*" in curr:
                res.add(curr["*"])
        else:
            return
        
        board[y][x] = "."
        _find(curr, y, x - 1)
        _find(curr, y - 1, x)
        _find(curr, y, x + 1)
        _find(curr, y + 1, x)
        board[y][x] = letter
        #could also delete parent[letter], but you should implement trie as an object

    for y in range(h):
        for x in range(w):
            _find(trie, y, x)
    return list(res)

def findWords(board: list[list[str]], words: list[str]) -> list[str]:
    w, h = len(board[0]), len(board)
    trie = {}
    for word in words:
        curr = trie
        for c in word:
            if c not in curr: curr[c] = {}
            curr = curr[c]
        curr["*"] = word
    
    res = []
    def _dfs(x, y, parent):
        char = board[y][x]
        curr = parent[char]

        word = curr.pop("*", False)
        if word: res.append(word)

        board[y][x] = "."
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w) or not (0 <= ny < h): continue
            
            nchar = board[ny][nx]
            if nchar == ".": continue
            if nchar not in curr: continue
            _dfs(nx, ny, curr)
        board[y][x] = char

        #if there are no subnodes in curr, there is no reason to check this node in parent
        if not curr: parent.pop(char) 

    for y in range(h):
        for x in range(w):
            if board[y][x] in trie: _dfs(x, y, trie)
    return list(res)  

#https://leetcode.com/problems/reverse-nodes-in-k-group/solutions/6896538/video-recursive-pattern-bonus-coding-with-iterative-pattern/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100           
def reverseKGroup(head: ListNode, k: int) -> ListNode:
    def _get(curr, k):
        while curr and k > 0:
            curr = curr.next
            k -= 1
        return curr
    
    l = res = ListNode(0, head)
    while True:
        r = _get(l, k)
        if not r: break

        prev, node = r.next, l.next
        while prev != r:
            temp = node.next
            node.next = prev
            prev = node
            node = temp

        temp = l.next 
        l.next = prev
        l = temp
    return res.next

def reverseKGroup(head: ListNode, k: int) -> ListNode:
    if not head:
        return None

    tail = head

    for _ in range(k):
        if not tail:
            return head
        tail = tail.next

    def reverse(cur, end):
        prev = None

        while cur != end:
            next = cur.next
            cur.next = prev
            prev = cur
            cur = next

        return prev      

    new_head = reverse(head, tail)
    head.next = reverseKGroup(tail, k)

    return new_head  

def reverseKGroup(head: ListNode, k: int) -> ListNode:
    def _reverse(node):
        prev = None
        while node:
            temp = node.next
            node.next = prev
            prev = node
            node = temp
        return prev
    
    res = phead = ListNode(0, None)
    l = r = head
    idx = 0
    while r:
        if idx == k - 1:
            idx = 0

            temp = r.next
            r.next = None
            phead.next = _reverse(l)
            while phead.next: phead = phead.next
            phead.next = temp
            l = r = temp
        else:
            idx = (idx + 1) % k
            r = r.next
    phead.next = l
    return res.next

#https://leetcode.com/problems/course-schedule-ii/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
from collections import defaultdict
def findOrder(numCourses: int, prerequisites: list[list[int]]) -> list[int]:
    req_to_c = defaultdict(list)
    req_num = [0]*numCourses
    for c, r in prerequisites:
        req_to_c[r].append(c)
        req_num[c] += 1

    order = []
    queue = []
    for i in range(numCourses):
        if not req_num[i]:
            queue.append(i)
    while queue:
        r = queue.pop(0)
        order.append(r)
        for c in req_to_c[r]:
            req_num[c] -= 1
            if not req_num[c]:
                queue.append(c)
                
    if len(order) == numCourses:
        return order
    else:
        return []
    
def findOrder(numCourses: int, prerequisites: list[list[int]]) -> list[int]:
    req = defaultdict(list)
    for c, r in prerequisites:
        req[c].append(r)

    order = []
    used = set()
    def _dfs(c):    
        nonlocal order
        if not req[c]:
            if c not in used:
                used.add(c)
                order = [c] + order
            return True
        if c in used:
            return False
        
        used.add(c)
        for r in req[c]:
            if not _dfs(r):
                return False
        order += [c]
        req[c] = []
        return True

    for i in range(numCourses):
        if not _dfs(i):
            return []
    return order

def findOrder(numCourses: int, prerequisites: list[list[int]]) -> list[int]:
    inc, outc = defaultdict(set), defaultdict(set)
    for c, r in prerequisites:
        inc[c].add(r)
        outc[r].add(c)

    q = collections.deque()
    for c in range(numCourses):
        if not inc[c]: q.append(c)

    res = []
    while q:
        qc = q.popleft()
        res.append(qc)
        for relc in outc[qc]:
            inc[relc] -= {qc}
            if not inc[relc]: q.append(relc)
    if all(not inc[c] for c in range(numCourses)): return res
    else: return []

def findOrder(numCourses: int, prerequisites: list[list[int]]) -> list[int]:
    req = defaultdict(set)
    for c, r in prerequisites: req[c].add(r)

    res = []
    visited, cycle = set(), set()
    def _dfs(c):
        if c in cycle: return False
        if c in visited: return True
        
        cycle.add(c)
        for rc in req[c]:
            if not _dfs(rc): return False
        cycle.remove(c)

        visited.add(c)
        res.append(c)
        return True
    
    for c in range(numCourses):
        if not _dfs(c): return []
    return res

#https://leetcode.com/problems/remove-element/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def removeElement(nums: list[int], val: int) -> int:
    if not nums:
        return 0

    i = 0
    while True:
        if nums[i] == val:
            del nums[i]
        else:
            i += 1
        if i == len(nums):
            return i
        
#https://leetcode.com/problems/rotate-array/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100    
def rotate(nums: list[int], k: int) -> None:
    axis = k % len(nums) - 1

    def _rev(i, j):
        while i < j:
            nums[i], nums[j] = nums[j], nums[i]
            j -= 1
            i += 1

    _rev(0, len(nums) - 1)
    _rev(0, axis)
    _rev(axis + 1, len(nums) - 1)

#https://leetcode.com/problems/bitwise-and-of-numbers-range/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def rangeBitwiseAnd(left: int, right: int) -> int:
    cnt = 0
    while left < right:
        left >>= 1
        right >>= 1
        cnt += 1
    return left << cnt

#https://leetcode.com/problems/palindrome-number/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def isPalindrome(x: int) -> bool:
    if x < 0:
        return False

    num = x
    mirror = 0
    while x:
        rem = x % 10
        mirror *= 10
        mirror += rem
        x //= 10
    return mirror == num

#https://leetcode.com/problems/plus-one/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def plusOne(digits: list[int]) -> list[int]:
    res = []
    rem = 1
    for i in range(len(digits) - 1, -1, -1):
        num = digits[i] + rem
        rem = num // 10
        res = [num % 10] + res
    if rem:
        res = [rem] + res
    return res

def plusOne(digits: list[int]) -> list[int]:
    res = digits.copy()
    carry = 1
    for i in range(len(res) - 1, -1, -1):
        sm = res[i] + carry
        carry, res[i] = sm // 10, sm % 10

    if carry: res = [carry] + res
    return res

#https://leetcode.com/problems/sqrtx/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def mySqrt(x: int) -> int:
    if x == 0:
        return 0
    
    res = 1
    def _find(l, r):
        if l < r:
            nonlocal x, res
            n = (l + r) // 2
            if x // n >= n:
                res = max(res, n)
                _find(n + 1, r)
            else:
                _find(l, n)
        else:
            return
    _find(1, x)
    return res

def mySqrt(x: int) -> int:
    sqrt = x
    while True:
        if sqrt * sqrt > x:
            sqrt = sqrt // 2
        else:
            if sqrt * sqrt <= x and (sqrt + 1) * (sqrt + 1) > x:
                return sqrt
            sqrt += 1

#https://www.youtube.com/watch?v=-OJJ78MbOLQ
def myPow(x: float, n: int) -> float:
    def _calc(x, n):
        if x == 0:
            return 0
        if n == 0:
            return 1
        
        res = _calc(x, n // 2)
        res *= res
        if n % 2:
            return res * x
        else:
            return res
        
    res = _calc(x, abs(n))
    if n >= 0:
        return res
    else:
        return 1 / res
    
def myPow(x: float, n: int) -> float:
    if x == 0 or x == 1: return x
    if n < 0: 
        x = 1 / x
        n = -n

    @cache
    def _dfs(n):
        if n == 0: return 1
        if n == 1: return x

        if n % 2 != 0: return x * _dfs(n - 1)
        else: return _dfs(n // 2) * _dfs(n // 2)
    return _dfs(n)

#https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100 
def buildTree(inorder: list[int], postorder: list[int]) -> TreeNode:
    inorder_idx = {val:idx for idx, val in enumerate(inorder)}
    def _build(l, r):
        if l > r:
            return None
        
        head = TreeNode(postorder.pop())
        idx = inorder_idx[head.val]
        head.right = _build(idx + 1, r)
        head.left = _build(l, idx - 1)
        return head
    return _build(0, len(inorder) - 1)

#https://leetcode.com/problems/path-sum/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def hasPathSum(root: TreeNode, targetSum: int) -> bool:
    def _calc(head, sub):
        if not head:
            return False
        if not head.left and not head.right:
            return head.val == sub
        return _calc(head.left, sub - head.val) or _calc(head.right, sub - head.val)
    return _calc(root, targetSum)

#https://leetcode.com/problems/binary-tree-right-side-view/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def rightSideView(root: TreeNode) -> int:
    view = []
    if not root:
        return view
    queue = [root]
    right = root.val
    #traverse layer before going to the next (slow)
    while queue: 
        view.append(right)
        for _ in range(len(queue)):
            head = queue.pop(0)
            if head.left:
                right = head.left.val
                queue.append(head.left)
            if head.right:
                right = head.right.val
                queue.append(head.right)
    return view

def rightSideView(root: TreeNode) -> list[int]:
    right = []
    queue = [(root, 0)]
    #traverse using queue (classic, fast)
    while queue:
        head, level = queue.pop(0)
        if not head:
            continue
        if len(right) == level:
            right.append(head.val)
        
        right[level] = head.val
        queue.append((head.left, level + 1))
        queue.append((head.right, level + 1))
    return right

#https://leetcode.com/problems/binary-tree-level-order-traversal/description/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def levelOrder(root: TreeNode) -> list[list[int]]:
    levels = []
    queue = [(root, 0)]
    while queue:
        node, level = queue.pop(0)
        if not node:
            continue
        if len(levels) == level:
            levels.append([])
        levels[level].append(node.val)
        queue.append((node.left, level + 1))
        queue.append((node.right, level + 1))
    return levels

#https://leetcode.com/problems/minimum-absolute-difference-in-bst/submissions/1764158521/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def getMinimumDifference(root: TreeNode) -> int:
    prev = None
    mn = float("inf")
    def _search(node):
        nonlocal mn, prev
        if node:
            _search(node.left)
            if prev != None:
                mn = min(mn, node.val - prev)
            prev = node.val
            _search(node.right)
    _search(root)
    return mn

def getMinimumDifference(root: TreeNode) -> int:
    def _traverse(node):
        if not node:
            return []
        return _traverse(node.left) + [node.val] + _traverse(node.right)
    
    inorder = _traverse(root)
    mn = float("inf")
    for i in range(1, len(inorder)):
        mn = min(mn, inorder[i] - inorder[i - 1])
    return mn

#https://leetcode.com/problems/surrounded-regions/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def solve(board: list[list[str]]) -> None:
    w = len(board[0])
    h = len(board)
    def _unsafe(y, x):
        if not (0 <= x < w) or not (0 <= y < h) or board[y][x] in (".", "X"):
            return

        board[y][x] = "."
        for dy, dx in [[-1, 0], [0, -1], [1, 0], [0, 1]]:
            _unsafe(y + dy, x + dx)
        return

    for y in range(h):
        _unsafe(y, 0)
        _unsafe(y, w - 1)
    for x in range(1, w - 1):
        _unsafe(0, x)
        _unsafe(h - 1, x)
    
    for y in range(h):
        for x in range(w):
            if board[y][x] == ".":
                board[y][x] = "O"
            elif board[y][x] == "O":
                board[y][x] = "X"

#BFS
def solve(board: list[list[str]]) -> None:
    w, h = len(board[0]), len(board)
    def _bfs(x, y, repl):
        q = collections.deque([(x, y)])
        chk = set([(x, y)])
        while q:
            x, y = q.popleft()
            board[y][x] = repl
            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w) or not (0 <= ny < h) or (board[ny][nx] == "X") or (nx, ny) in chk: continue
                q.append((nx, ny))
                chk.add((nx, ny))

    for x in range(w):
        if board[0][x] == "O": _bfs(x, 0, "N")
        if board[h - 1][x] == "O": _bfs(x, h - 1, "N")
    for y in range(h):
        if board[y][0] == "O": _bfs(0, y, "N")
        if board[y][w - 1] == "O": _bfs(w - 1, y, "N")
    
    for y in range(h):
        for x in range(w):
            if board[y][x] == "O": board[y][x] = "X"
            elif board[y][x] == "N": board[y][x] = "O"

#DSU, pretty useless tbh
def solve(board: list[list[str]]) -> None:
    w, h = len(board[0]), len(board)
    def _idx(x, y): return y * w + x

    p = list(range(w * h + 1))
    ppwr = [1] * (w * h + 1)
    ppwr[w * h] = float("inf")
    def _find(x):
        if p[x] == x: return x
        else:
            p[x] = _find(p[x])
            return p[x]
    
    for y in range(h):
        for x in range(w):
            if board[y][x] != "O": continue
            if (x == 0) or (x == w - 1) or (y == 0) or (y == h - 1):
                pa = _find(_idx(x, y))
                p[pa] = w * h
                ppwr[pa] -= 1
                ppwr[w * h] += 1  
            else:
                pa = _find(_idx(x, y))
                for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if board[ny][nx] == "O":
                        pb = _find(_idx(nx, ny))
                        if pa == pb: continue
                        if ppwr[pa] >= ppwr[pb]: 
                            p[pb] = pa
                            ppwr[pa] += ppwr[pb]
                            ppwr[pb] = 0
                        else:
                            p[pa] = pb
                            ppwr[pb] += ppwr[pa]
                            ppwr[pa] = 0
    for y in range(h):
        for x in range(w):
            if board[y][x] == "O" and _find(_idx(x, y)) != w * h:
                board[y][x] = "X"

#https://leetcode.com/problems/clone-graph/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def cloneGraph(node: GraphNode) -> GraphNode:
    if not node:
        return node
    
    copied = {
        node.val: GraphNode(node.val, [])
    }
    queue = [node]
    while queue:
        el = queue.pop(0)
        c_el = copied[el.val]

        for n in el.neighbors:
            if n.val not in copied:
                copied[n.val] = Node(n.val, [])
                queue.append(n)
            c_el.neighbors.append(copied[n.val])
    return copied[node.val]

def cloneGraph(node: GraphNode):
    nodes = {}
    def _dfs(curr):
        repl = Node(curr.val)
        nodes[curr.val] = repl
        for n in curr.neighbors:
            if n.val not in nodes: _dfs(n)
            repl.neighbors.append(nodes[n.val])
        return repl
    return _dfs(node) if node else None

#https://leetcode.com/problems/evaluate-division/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def calcEquation(equations: list[list[str]], values: list[float], queries: list[list[str]]) -> list[float]:
    nodes = defaultdict(dict)
    for i in range(len(equations)):
        eq = equations[i]
        nodes[eq[0]][eq[1]] = values[i]
        nodes[eq[1]][eq[0]] = 1 / values[i]

    def _dfs(node, target, visited):
        visited.append(node)
        if not nodes.get(node, False):
            return None
        if target in nodes[node]:
            return nodes[node].get(target, None)
        
        for path in nodes[node]:
            if nodes[node][path] < 0:
                continue
            if path not in visited:
                val = _dfs(path, target, visited) 
                if val:
                    return val * nodes[node][path]
        return None
    
    res = [0] * len(queries)
    for i in range(len(queries)):
        n1, n2 = queries[i]
        res[i] = _dfs(n1, n2, []) or -1.0
        nodes[n1][n2] = res[i]
            
    return res

#https://leetcode.com/problems/generate-parentheses/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def generateParenthesis(n: int) -> list[str]:
        basic = {
            1: {"()"},
            2: {"()()", "(())"}
        }

        for i in range(3, n + 1):
            basic[i] = set()
            for j in range(1, i):
                for prefix in basic[j]:
                    for suffix in basic[i - j]:                                                                                                                   
                        if j == 1:
                            basic[i].add("(" + suffix + ")")
                        basic[i].add(prefix + suffix)
        return list(basic[n])

def generateParenthesis(n: int) -> list[str]:
    res = []
    def _create(curr, openP = 0, closedP = 0):
        if openP == closedP and openP + closedP == n * 2:
            res.append(curr)
            return
        
        if openP < n:
            _create(curr + "(", openP + 1, closedP)
        
        if openP > closedP:
            _create(curr + ")", openP, closedP + 1)
        
    _create("")
    return res

#https://leetcode.com/problems/sort-list/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def sortList(head: ListNode) -> ListNode:
    if not head or not head.next:
        return head
    def _node_merge(n1, n2):
        head = dum = ListNode()
        while n1 and n2:
            if n1.val <= n2.val:
                head.next = n1 
                n1 = n1.next
            else:
                head.next = n2 
                n2 = n2.next
            head = head.next

        if n1:
            head.next = n1
        else:
            head.next = n2
        return dum.next

    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    mid = slow.next #actually, element after mid
    slow.next = None

    l = sortList(head)
    r = sortList(mid)
    return _node_merge(l, r)

#https://leetcode.com/problems/maximum-subarray/
#⭐
def maxSubArray(nums: list[int]) -> int:
    dp = [0] * len(nums)
    dp[0] = nums[0]
    for i in range(1, len(nums)):
        dp[i] = max(nums[i] + dp[i - 1], nums[i])
    return max(dp)

#Kadane's algorithm
def maxSubArray(nums: list[int]) -> int:
    curr = mx = nums[0]
    for n in nums[1:]:
        curr = max(curr + n, n)
        mx = max(curr, mx)
    return mx

#https://leetcode.com/problems/maximum-sum-circular-subarray/solutions/178422/one-pass/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
#I didn't understand problem, just copied solution
def maxSubarraySumCircular(nums: list[int]) -> int:
    total = nums[0]
    mx = currMx = nums[0]
    mn = currMn = nums[0]

    for n in nums[1:]:
        currMx = max(currMx + n, n)
        currMn = min(currMn + n, n)
        mx = max(mx, currMx)
        mn = min(mn, currMn)
        total += n
    return max(mx, total - mn) if mx > 0 else mx

#https://leetcode.com/problems/find-peak-element/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def findPeakElement(nums: list[int]) -> int:
    l = 0
    r = len(nums) - 1
    mx = [float("-inf"), 0]
    while l <= r:
        m = (l + r) // 2
        if m < len(nums) - 1 and nums[m] < nums[m + 1]:
            l = m + 1
        elif m > 0 and nums[m] < nums[m - 1]:
            r = m - 1
        else:
            if nums[m] > mx[0]:
                mx = [nums[m], m]
            break
    return mx[1]

def findPeakElement(nums: list[int]) -> int:
    l = 0
    r = len(nums) - 1
    while l < r:
        m = (l + r) // 2
        if nums[m] > nums[m + 1]:
            r = m
        else:
            l = m + 1
    return l

#https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def findMin(nums: list[int]) -> int:
    l = 0
    r = len(nums) - 1
    while l < r:
        mid = (l + r) // 2
        if nums[mid] > nums[mid + 1]:
            l = mid + 1
        else:
            if nums[-1] > nums[0]:
                r = mid
            elif nums[mid] > nums[-1]:
                l = mid + 1
            else:
                r -= 1
    return nums[l]

def findMin(nums: list[int]) -> int:
    l = 0
    r = len(nums) - 1
    while l < r:
        mid = (l + r) // 2
        if nums[mid] <= nums[r]:
            r = mid
        else:
            l = mid + 1
    return nums[l]

#my solution
def findMin(nums: list[int]) -> int:
    l, r = 0, len(nums) - 1
    mn = float("inf")
    while l <= r:
        m = l + ((r - l) // 2)
        mn = min(nums[m], mn)
        if nums[m] >= nums[r]: l = m + 1 #first half
        elif nums[m] <= nums[l]: r = m - 1 #second half
        elif nums[l] <= nums[m] <= nums[r]: r = m - 1 #sorted
    return mn

#https://leetcode.com/problems/remove-duplicates-from-sorted-array/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def removeDuplicates(nums: list[int]) -> int:
    focus = nums[0] #my solution is better, because it actually removes sequential duplicate elements Xd
    i = 1 
    while i < len(nums):
        if nums[i] == focus:
            del nums[i]
        else:
            focus = nums[i]
            i += 1
    return i

def removeDuplicates(nums: list[int]) -> int:
    if not nums:
        return 0

    j = 1
    for i in range(1, len(nums)):
        if nums[i] != nums[i - 1]:
            nums[j] = nums[i]
            j += 1
    return j

#https://leetcode.com/problems/merge-strings-alternately/?envType=study-plan-v2&envId=leetcode-75
def mergeAlternately(word1: str, word2: str) -> str:
    s = ""
    i = j = 0
    while i < len(word1) and j < len(word2):
        s += word1[i] + word2[j]
        i += 1
        j += 1
    if i < len(word1):
        s += word1[i:]
    else:
        s += word2[j:]
    
    return s

def mergeAlternately(word1: str, word2: str) -> str:
    s = []
    for c1, c2 in zip(word1, word2):
        s.append(c1 + c2)
    s.append(word1[len(word2):])
    s.append(word2[len(word1):])
    return "".join(s)

#https://leetcode.com/problems/string-compression/?envType=study-plan-v2&envId=leetcode-75
def compress(chars: list[str]) -> int:
    focus = chars[0] #my solution is correct yet again, because it actually compresses string
    cnt = 1
    i = 1

    while i < len(chars):
        if chars[i] != focus:
            focus = chars[i]
            if cnt > 1:
                k = i
                while cnt:
                    chars.insert(k, str(cnt % 10))
                    cnt //= 10
                    i += 1
                cnt = 1
            i += 1
        else:
            del chars[i]
            cnt += 1
    if cnt > 1:
        k = i
        while cnt:
            chars.insert(i, str(cnt % 10))
            cnt //= 10
    return len(chars)

def compress(chars: list[str]) -> int:
    focus = 0
    i = 0
    while i < len(chars):
        j = i
        while j < len(chars) and chars[j] == chars[i]:
            j += 1
        l = j - i
        if l > 1:
            sl = str(l)
            chars[focus] = chars[i]
            for c in sl:
                focus += 1
                chars[focus] = c
        else:
            chars[focus] = chars[i]
        focus += 1
        i = j
    return focus

#https://leetcode.com/problems/increasing-triplet-subsequence/?envType=study-plan-v2&envId=leetcode-75
def increasingTriplet(nums: list[int]) -> bool:
    mn = mn1 = float("inf")
    for i in range(len(nums)):
        if mn < mn1 < nums[i]:
            return True
        if nums[i] < mn:
            mn = nums[i]
        if mn < nums[i] < mn1:
            mn1 = nums[i]
    return False

def increasingTriplet(nums: list[int]) -> bool:
    mn = mn1 = float("inf")
    for i in range(len(nums)):
        if nums[i] <= mn:
            mn = nums[i]
        elif nums[i] <= mn1:
            mn1 = nums[i]
        else:
            return True
    return False

#https://leetcode.com/problems/reverse-words-in-a-string/?envType=study-plan-v2&envId=leetcode-75
def reverseWords(s: str) -> str:
    w = s.split()
    l = 0
    r = len(w) - 1
    while l < r:
        w[l], w[r] = w[r], w[l]
        l += 1
        r -= 1
    return " ".join(w)

def reverseWords(s: str) -> str: #medium problem btw
    s = s.split()
    return " ".join(s[::-1])

#https://leetcode.com/problems/reverse-vowels-of-a-string/?envType=study-plan-v2&envId=leetcode-75
def reverseVowels(s: str) -> str:
    v = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'} #works faster with set
    res = list(s)
    i = 0
    j = len(res) - 1
    while i < j:
        if res[i] not in v:
            i += 1
        elif res[j] not in v:
            j -= 1
        else:
            res[i], res[j] = res[j], res[i]
            i += 1
            j -= 1

    return "".join(res)

#https://leetcode.com/problems/can-place-flowers/?envType=study-plan-v2&envId=leetcode-75
def canPlaceFlowers(flowerbed: list[int], n: int) -> bool:
    for i in range(len(flowerbed)):
        l = flowerbed[i - 1] if i > 0 else flowerbed[i]
        r = flowerbed[i + 1] if i < len(flowerbed) - 1 else flowerbed[i]
        if l == flowerbed[i] == r == 0:
            n -= 1
            flowerbed[i] = 1
    return n <= 0

def canPlaceFlowers(flowerbed: list[int], n: int) -> bool:
    cnt = 1 
    for i in range(len(flowerbed)):
        if flowerbed[i] == 0:
            cnt += 1
        else:
            cnt = 0

        if cnt == 3:
            n -= 1
            cnt = 1
            if n == 0:
                return True
    if cnt == 2:
        n -= 1
    return n <= 0

#https://leetcode.com/problems/kids-with-the-greatest-number-of-candies/?envType=study-plan-v2&envId=leetcode-75
def kidsWithCandies(candies: list[int], extraCandies: int) -> list[bool]:
    res = [False] * len(candies)
    for i in range(len(candies)):
        if candies[i] + extraCandies >= max(candies):
            res[i] = True
    return res

#https://leetcode.com/problems/greatest-common-divisor-of-strings/?envType=study-plan-v2&envId=leetcode-75
def gcdOfStrings(str1: str, str2: str) -> str:
    from math import gcd
    slen1, slen2 = len(str1), len(str2)
    mxlen = gcd(slen1, slen2)
    sub = str1[:mxlen]
    if sub * (slen1 // mxlen) == str1 and sub * (slen2 // mxlen) == str2:
        return sub
    else:
        return ""

#https://leetcode.com/problems/domino-and-tromino-tiling/?envType=study-plan-v2&envId=leetcode-75  
def numTilings(n: int) -> int:
    memo = {-1: 0, 0: 1, 1: 1, 2: 2, 3: 5}
    mod = 10**9 + 7
    for i in range(4, n + 1):
        memo[i] = memo[i - 3] + memo[i - 1] * 2 #ex: 5 = 1 + 2*2; 11 = 5*2 + 1
    return memo[n] % mod

def numTilings(n: int) -> int:
    memo = [0, 1, 2, 5] + [0] * 997
    mod = 10**9 + 7
    for i in range(4, n + 1):
        memo[i] = memo[i - 3] + memo[i - 1] * 2
    return memo[n] % mod

#https://leetcode.com/problems/n-th-tribonacci-number/?envType=study-plan-v2&envId=leetcode-75
def tribonacci(n: int) -> int:
    seq = [0, 1, 1]
    if n <= 2:
        return seq[n]
    for _ in range(3, n + 1):
        nxt = sum(seq)
        seq.pop(0)
        seq.append(nxt)
    return seq[-1]

#https://leetcode.com/problems/min-cost-climbing-stairs/?envType=study-plan-v2&envId=leetcode-75
def minCostClimbingStairs(cost: list[int]) -> int:
    window = [cost[0], cost[1]]
    for i in range(2, len(cost)):
        nxt = cost[i] + min(window)
        window.pop(0)
        window.append(nxt)
    return min(window)

def minCostClimbingStairs(cost: list[int]) -> int:
    memo = [0] * len(cost)
    memo[0], memo[1] = cost[0], cost[1]
    for i in range(2, len(cost)):
        memo[i] = cost[i] + min(memo[i - 1], memo[i - 2])
    return min(memo[-1], memo[-2])

def minCostClimbingStairs(cost: list[int]) -> int:
    if len(cost) <= 2: return min(cost)

    n2, n1 = cost[0], cost[1]
    curr = 0
    for i in range(2, len(cost)):
        curr = cost[i] + min(n2, n1)
        n2, n1, curr = n1, curr, 0
    return min(n1, n2)

#https://leetcode.com/problems/edit-distance/?envType=study-plan-v2&envId=leetcode-75
#⭐o algo
def minDistance(word1: str, word2: str) -> int:    
    m = len(word1) + 1
    n = len(word2) + 1
    memo = [[i] + [0] * (m - 1) for i in range(n)]
    for i in range(1, m):
        memo[0][i] = i

    for y in range(1, n):
        for x in range(1, m):
            if word2[y - 1] == word1[x - 1]:
                memo[y][x] = memo[y - 1][x - 1]
            else:
                memo[y][x] = 1 + min(memo[y - 1][x], memo[y][x - 1], memo[y - 1][x - 1])
    return memo[-1][-1]

def minDistance(word1: str, word2: str) -> int:    
    r1 = [0] * (len(word2) + 1)
    for i in range(1, len(word2) + 1): r1[i] = i

    for j in range(len(word1)):
        r2 = [j + 1] + [0] * len(word2)
        for i in range(1, len(word2) + 1):   
            if word2[i - 1] == word1[j]: r2[i] = r1[i - 1]
            else: r2[i] = 1 + min(r2[i - 1], r1[i - 1], r1[i])
        r1 = r2
    return r1[-1]

#https://leetcode.com/problems/longest-common-subsequence/?envType=study-plan-v2&envId=leetcode-75
def longestCommonSubsequence(text1: str, text2: str) -> int:
    l1 = len(text1) + 1
    l2 = len(text2) + 1
    memo = [0] * l1
    for i in range(1, l2):
        sub = memo[::]
        for j in range(1, l1):
            if text1[j - 1] == text2[i - 1]:
                sub[j] = 1 + memo[j - 1]
                #sub[j] = 1 + min(sub[j - 1], memo[j - 1], memo[j])
            else:
                sub[j] = max(sub[j - 1], memo[j - 1], memo[j])
        memo = sub
    return memo[-1]

def longestCommonSubsequence(text1: str, text2: str) -> int:
    r1 = [0] * (len(text1) + 1)
    for i in range(len(text2)):
        r2 = [0] * (len(text1) + 1)
        for j in range(1, len(text1) + 1):
            if text2[i] == text1[j - 1]: r2[j] = 1 + r1[j - 1]
            else: r2[j] = max(r2[j - 1], r1[j])
        r1 = r2
    return r1[-1]

#https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/?envType=study-plan-v2&envId=leetcode-75
def maxProfit(prices: list[int], fee: int) -> int:
    wS = -1 * prices[0]
    wtS = 0
    for p in prices:
        wS = max(wS, wtS - p)
        wtS = max(wtS, p + wS - fee) #wS is initially negative
    return wtS

#https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/?envType=study-plan-v2&envId=leetcode-75
def findMinArrowShots(points: list[list[int]]) -> int:
    points = sorted(points, key=lambda x: x[0])
    cnt = 0
    curr = [float("-inf"), float("-inf")]
    for start, end in points:
        if start > curr[1]:
            curr = [start, end]
            cnt += 1
        else:
            if start > curr[0]:
                curr[0] = start
            if end < curr[1]:
                curr[1] = end
    return cnt

def findMinArrowShots(points: list[list[int]]) -> int:
    points = sorted(points, key=lambda x: x[1])
    cnt = 1
    pos = points[0][1]
    for start, end in points[1:]:
        if start > pos:
            pos = end
            cnt += 1
    return cnt

#https://leetcode.com/problems/non-overlapping-intervals/?envType=study-plan-v2&envId=leetcode-75
def eraseOverlapIntervals(intervals: list[list[int]]) -> int:
    intervals = sorted(intervals, key=lambda x: x[1])
    cnt = 0
    curr_end = intervals[0][1]
    for start, end in intervals[1:]:
        if start < curr_end:
            cnt += 1
        else:
            curr_end = end
    return cnt

def eraseOverlapIntervals(intervals: list[list[int]]) -> int:
    t = sorted(intervals)
    k = 0
    curre = t[0][1]
    for s, e in t[1:]:
        if s < curre: 
            curre = min(curre, e) #los greedy algorithm o algo
            k += 1
        else: curre = e #next interval
    return k

#https://leetcode.com/problems/online-stock-span/?envType=study-plan-v2&envId=leetcode-75
class StockSpanner:

    def __init__(self):
        self.stack = []

    def next(self, price: int) -> int:
        span = 1
        while self.stack and self.stack[-1][0] <= price:
            span += self.stack.pop()[1]
        self.stack.append([price, span])
        return span 
    
#https://leetcode.com/problems/daily-temperatures/?envType=study-plan-v2&envId=leetcode-75
def dailyTemperatures(temperatures: list[int]) -> list[int]:
    stack = []
    res = [0] * len(temperatures)
    for i in range(len(temperatures)):
        while stack and temperatures[stack[-1]] < temperatures[i]:
            idx = stack.pop()
            res[idx] = i - idx
        stack.append(i)
    return res

#⭐https://leetcode.com/problems/search-suggestions-system/?envType=study-plan-v2&envId=leetcode-75
def suggestedProducts(products: list[str], searchWord: str) -> list[list[str]]:
    class TrieNode:
        def __init__(self):
            self.children = defaultdict(TrieNode)
            self.suggestions = []
        
        def add_suggestion(self, suggestion):
            if len(self.suggestions) < 3:
                self.suggestions.append(suggestion)

    products = sorted(products)
    trie = TrieNode()
    for w in products:
        curr = trie
        for c in w:
            curr = curr.children[c]
            curr.add_suggestion(w)

    res = []
    curr = trie
    for c in searchWord:
        curr = curr.children[c]
        res.append(curr.suggestions)
    return res

def suggestedProducts(products: list[str], searchWord: str) -> list[list[str]]:
    products = sorted(products)
    res = []
    l = 0
    r = len(products) - 1
    for i in range(len(searchWord)):
        c = searchWord[i]
        while l <= r and (len(products[l]) < i + 1 or products[l][i] != c): #i + 1: length of a prefix
            l += 1
        while l <= r and (len(products[r]) < i + 1 or products[r][i] != c):
            r -= 1
        
        sub = []
        rem = r - l + 1
        for j in range(min(rem, 3)):
            sub.append(products[l + j])
        res.append(sub)
    return res

#https://leetcode.com/problems/successful-pairs-of-spells-and-potions/?envType=study-plan-v2&envId=leetcode-75
def successfulPairs(spells: list[int], potions: list[int], success: int) -> list[int]:
    def _search(s):
        l = 0
        r = len(potions) - 1
        idx = -1
        while l <= r:
            mid = (l + r) // 2
            if potions[mid] * s >= success:
                idx = mid
                r = mid - 1
            else:
                l = mid + 1
        return idx

    potions = sorted(potions)
    res = []
    for s in spells:
        idx = _search(s)
        if idx == -1:
            res.append(0)
        else:
            res.append(len(potions) - idx)
    return res

#https://leetcode.com/problems/koko-eating-bananas/?envType=study-plan-v2&envId=leetcode-75
def minEatingSpeed(piles: list[int], h: int) -> int:
    from math import ceil
    def _check(k):
        total = 0
        for n in piles:
            total += ceil(n / k)
        return total <= h
    
    l = 1
    r = max(piles)
    while l <= r:
        m = (l + r) // 2
        if _check(m):
            r = m - 1
        else:
            l = m + 1
    return l

def minEatingSpeed(piles: list[int], h: int) -> int:
    def _check(k):
        total = 0
        for p in piles:
            total += ceil(p / k)
        return total
    
    l, r = 1, max(piles) + 1
    res = float("inf")
    while l <= r:
        m = l + ((r - l) // 2)
        total = _check(m)
        if total > h: l = m + 1
        elif total <= h:
            res = min(m, res)
            r = m - 1
    return res

#https://leetcode.com/problems/guess-number-higher-or-lower/?envType=study-plan-v2&envId=leetcode-75
def guessNumber(n: int) -> int:
    l = 1
    r = n 
    while l <= r:
        m = (l + r) // 2
        g = "guess(m)" # def guess(num: int) -> int:
        if g == 0:
            return m
        elif g == 1:
            l = m + 1
        else:
            r = m - 1
    return l

def totalCost(costs: list[int], k: int, candidates: int) -> int:
    from heapq import heappush, heappop

    sm = 0
    la, ra = [], []
    l, r = 0, len(costs) - 1
    while k > 0:
        while len(la) < candidates and l <= r:
            heappush(la, costs[l])
            l += 1
        while len(ra) < candidates and l <= r:
            heappush(ra, costs[r])
            r -= 1
        
        t1 = la[0] if la else float("inf")
        t2 = ra[0] if ra else float("inf")

        if t1 <= t2:
            sm += heappop(la)
        else:
            sm += heappop(ra)
        k -= 1
    return sm


def totalCost(costs: list[int], k: int, candidates: int) -> int:
    from heapq import heappush, heappop, heapify

    res = 0 
    la = costs[:candidates]
    ra = costs[max(candidates, len(costs) - candidates):]
    heapify(la)
    heapify(ra)
    l, r = candidates, len(costs) - candidates - 1 #next l and next r
    for _ in range(k):
        if (not ra) or (la and la[0] <= ra[0]):
            res += heappop(la)
            if l <= r:
                heappush(la, costs[l])
                l += 1
        else:
            res += heappop(ra)
            if l <= r:
                heappush(ra, costs[r])
                r -= 1
    return res

#https://leetcode.com/problems/smallest-number-in-infinite-set/?envType=study-plan-v2&envId=leetcode-75
class SmallestInfiniteSet:
    def __init__(self):
        self.heap = [1]
        self.mx = 0

    def popSmallest(self) -> int:
        sm = heappop(self.heap)
        if sm + 1 > self.mx:
            heappush(self.heap, sm + 1)
            self.mx = sm + 1
        return sm

    def addBack(self, num: int) -> None:
        if num < self.mx and num not in self.heap:
            heappush(self.heap, num)

#https://leetcode.com/problems/maximum-subsequence-score/?envType=study-plan-v2&envId=leetcode-75
def maxScore(nums1: list[int], nums2: list[int], k: int) -> int:
    s = sorted(zip(nums2, nums1), reverse=True)
    heap = []
    sm = 0
    res = 0
    
    for i in range(len(s)):
        if len(heap) < k:
            sm += s[i][1]
            heappush(heap, s[i][1])
        else:
            res = max(res, sm * s[i - 1][0])
            sm -= heappushpop(heap, s[i][1])
            sm += s[i][1]
            
    return max(res, sm * s[i][0])

#https://leetcode.com/problems/delete-node-in-a-bst/?envType=study-plan-v2&envId=leetcode-75
def deleteNode(root: TreeNode, key: int) -> TreeNode:
    def _f(node, key):
        if not node:
            return None

        if node.val == key:
            if not node.left or not node.right:
                return node.left or node.right
            else:
                t = node.right
                while t.left:
                    t = t.left
                node.val = t.val
                node.right = _f(node.right, node.val)
        elif node.val > key:
            node.left = _f(node.left, key)
        else:
            node.right = _f(node.right, key)
        return node
    return _f(root, key)

#https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree/?envType=study-plan-v2&envId=leetcode-75
def maxLevelSum(root: TreeNode) -> int:
    mx = float("-inf")
    res = 0
    lvl = 0
    queue = collections.deque([root])
    while queue:
        lvl += 1
        sm = 0
        for _ in range(len(queue)):
            node = queue.popleft()
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
            sm += node.val

        if sm > mx:
            mx = sm
            res = lvl
    return res

#https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/?envType=study-plan-v2&envId=leetcode-75
def lowestCommonAncestor(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    common = None
    def _dfs(node):
        nonlocal common
        if not node: return None
        
        l = _dfs(node.left)
        r = _dfs(node.right)
        if node == p or node == q:
            if l or r:
                common = node
            return node
        elif l and r:
            common = node

        return l or r
    _dfs(root)
    return common

def lowestCommonAncestor(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    def _dfs(node):
        if not node or node == q or node == p: #if the first node is higher than second node, then we return the first node
            return node
        l = _dfs(node.left)
        r = _dfs(node.right)
        return node if l and r else l or r
    return _dfs(root)

def longestZigZag(root: TreeNode) -> int:
    mx = 0
    def _dfs(node, depth, direction):
        nonlocal mx
        if not node: return

        mx = max(mx, depth)
        if direction == 1:
            _dfs(node.left, depth + 1, 0)
            _dfs(node.right, 1, 1)
        else:
            _dfs(node.right, depth + 1, 1)
            _dfs(node.left, 1, 0)
    _dfs(root.right, 1, 1)
    _dfs(root.left, 1, 0)
    return mx

#https://leetcode.com/problems/subarray-sum-equals-k/
#⭐
def subarraySum(nums: list[int], k: int) -> int:
    totals = {0: 1}
    cnt = 0
    curr = 0
    for n in nums:
        curr += n
        if curr - k in totals:
            cnt += totals[curr - k]
        totals[curr] = 1 + totals.get(curr, 0)
    return cnt

#https://leetcode.com/problems/path-sum-iii/?envType=study-plan-v2&envId=leetcode-75
def pathSum(root: TreeNode, targetSum: int) -> int:
    k = 0
    totals = {0: 1}
    def _dfs(node, curr):
        nonlocal k
        if not node: return 
        
        curr += node.val
        if curr - targetSum in totals:
            k += totals[curr - targetSum]

        totals[curr] = 1 + totals.get(curr, 0)
        _dfs(node.left, curr)
        _dfs(node.right, curr)
        totals[curr] -= 1 #we go back, so it decrements

    _dfs(root, 0)
    return k

#https://leetcode.com/problems/count-good-nodes-in-binary-tree/?envType=study-plan-v2&envId=leetcode-75
def goodNodes(root: TreeNode) -> int:
    k = 0
    def _dfs(mx, node):
        nonlocal k
        if not node: return -inf

        if node.val >= mx:
            mx = node.val
            k += 1

        _dfs(mx, node.left)
        _dfs(mx, node.right)

        return node.val
    _dfs(-inf, root)
    return k

def goodNodes(root: TreeNode) -> int:
    k = 0
    def _dfs(mxprev, head):
        if not head: return 

        nonlocal k
        if head.val >= mxprev: k += 1
        _dfs(max(mxprev, head.val), head.left)
        _dfs(max(mxprev, head.val), head.right)
    _dfs(float("-inf"), root)
    return k

def goodNodes(root: TreeNode) -> int:
    k = 0
    q = collections.deque([(root, float("-inf"))])
    while q:
        node, mxval = q.pop()
        if node.val >= mxval:
            k += 1
            mxval = node.val
        
        if node.left: q.append((node.left, mxval))
        if node.right: q.append((node.right, mxval))
    return k

#https://leetcode.com/problems/dota2-senate/?envType=study-plan-v2&envId=leetcode-75
def predictPartyVictory(senate: str) -> str:
    stack = collections.deque(senate)
    leading = stack.popleft()
    moves = 1
    while stack:
        el = stack.popleft()
        if moves and el != leading:
            stack.append(leading)
            moves -= 1
        elif el == leading:
            moves += 1
        else:
            leading = el
            moves = 1     
    return "Radiant" if leading == "R" else "Dire"

#https://leetcode.com/problems/decode-string/?envType=study-plan-v2&envId=leetcode-75
def decodeString(s: str) -> str:
    def _decode(i):
        atom = ""
        repeat = 0
        while i < len(s):
            if s[i] == "[": 
                matom, end = _decode(i + 1)
                atom += matom * repeat
                i = end
                repeat = 0
            elif s[i] == "]":
                break
            elif s[i].isdigit():
                repeat *= 10
                repeat += int(s[i])
            else:
                atom += s[i]
            i += 1
        return (atom, i)
    return _decode(0)[0]

def decodeString(s: str) -> str:
    repeat = 0
    stack = [""]
    for i in range(len(s)):
        if s[i].isdigit():
            repeat *= 10
            repeat += int(s[i])
        elif s[i] == "[":
            stack.append(repeat)
            stack.append("")
            repeat = 0
        elif s[i] == "]":
            atom = stack.pop()
            arepeat = stack.pop()
            aprefix = stack.pop()
            stack.append(aprefix + atom * arepeat)
        else:
            stack[-1] += s[i]
    return "".join(stack)

#https://leetcode.com/problems/asteroid-collision/?envType=study-plan-v2&envId=leetcode-75
def asteroidCollision(asteroids: list[int]) -> list[int]:
    res = []
    for i in range(len(asteroids)):
        if asteroids[i] < 0:
            res.append(asteroids[i])
        else:
            break
    if len(res) == len(asteroids):
        return res 
    
    stack = collections.deque()
    for a in asteroids[i:]:
        if stack and stack[-1] > 0 and a < 0:
            while stack:
                p = stack.pop()
                if abs(a) > abs(p):
                    if not stack:
                        res.append(a)
                        break
                    continue 
                elif abs(a) == abs(p):
                    break
                else:
                    stack.append(p)
                    break
        else: 
            if not stack and a < 0: res.append(a)
            else: stack.append(a)

    return res + list(stack)

def asteroidCollision(asteroids: list[int]) -> list[int]:
    stack = collections.deque()
    for a in asteroids:
        while stack and stack[-1] > 0 and a < 0:
            p = stack.pop()
            diff = p + a
            if diff > 0:
                stack.append(p)
                a = 0
            elif diff < 0:
                continue
            else:
                a = 0
        if a != 0:
            stack.append(a)
    return list(stack)

#https://leetcode.com/problems/removing-stars-from-a-string/?envType=study-plan-v2&envId=leetcode-75
def removeStars(s: str) -> str:
    stack = []
    for c in s:
        if c != "*":
            stack.append(c)
        else:
            stack.pop()
    return "".join(stack)

#https://leetcode.com/problems/equal-row-and-column-pairs/?envType=study-plan-v2&envId=leetcode-75
def equalPairs(grid: list[list[int]]) -> int:
    cnt = 0
    rows = {}
    for r in grid:
        sr = tuple(r)
        rows[sr] = 1 + rows.get(sr, 0)
    
    for x in range(len(grid[0])):
        curr = []
        for y in range(len(grid)):
            curr.append(grid[y][x])
        cnt += rows.get(tuple(curr), 0)
    return cnt

#https://leetcode.com/problems/determine-if-two-strings-are-close/?envType=study-plan-v2&envId=leetcode-75         
def closeStrings(word1: str, word2: str) -> bool:
    if len(word1) != len(word2): return False
    if set(word1) != set(word2): return False
    
    w1 = sorted(collections.Counter(word1).values())
    w2 = sorted(collections.Counter(word2).values())
    return w1 == w2

#https://leetcode.com/problems/longest-subarray-of-1s-after-deleting-one-element/?envType=study-plan-v2&envId=leetcode-75
def longestSubarray(nums: list[int]) -> int:
    prev = -1 #in case nums[0] is 1
    sm = 0
    mx = 0
    for i in range(len(nums)):
        if nums[i]:
            sm += nums[i]
        else:
            prev = sm
            sm = 0
        mx = max(prev + sm, mx)
    return mx

#https://leetcode.com/problems/max-consecutive-ones-iii/?envType=study-plan-v2&envId=leetcode-75
def longestOnes(nums: list[int], k: int) -> int:
    mx = 0
    l = 0
    zeroes = 0
    for r in range(len(nums)):
        if nums[r] == 0:
            zeroes += 1
        if zeroes > k: 
            while zeroes > k:
                if nums[l] == 0:
                    zeroes -= 1
                l += 1
        mx = max(mx, r - l + 1)
    return mx

def longestOnes(nums: list[int], k: int) -> int:
    window = collections.deque()
    mx = 0
    for n in nums:
        window.append(n)
        if not n: k -= 1
        if k < 0:
            k += 1 - window.popleft()
        mx = max(len(window), mx)
    return mx

#https://leetcode.com/problems/simple-bank-system/?envType=daily-question&envId=2025-10-26
class Bank:

    def __init__(self, balance: list[int]):
        self.balance = balance
        self.accounts = len(self.balance) + 1
    
    def _exists(self, number):
        return (1 <= number <= self.accounts)

    def transfer(self, account1: int, account2: int, money: int) -> bool:
        if self._exists(account1) and self._exists(account2) and money <= self.balance[account1 - 1]:
            self.balance[account2 - 1] += money
            self.balance[account1 - 1] -= money
            return True
        else: return False

    def deposit(self, account: int, money: int) -> bool:
        if self._exists(account):
            self.balance[account - 1] += money
            return True
        else: return False

    def withdraw(self, account: int, money: int) -> bool:
        if self._exists(account) and money <= self.balance[account - 1]:
            self.balance[account - 1] -= money
            return True
        else: return False

#https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/?envType=study-plan-v2&envId=leetcode-75
def maxVowels(s: str, k: int) -> int:
    v = set('aeiou')
    cnt = 0
    l = 0
    r = 0
    while r < k:
        if s[r] in v:
            cnt += 1
        r += 1
    
    mx = cnt
    while r < len(s):
        if s[r] in v:
            cnt += 1
        if s[l] in v:
            cnt -= 1
        l += 1
        r += 1
        mx = max(mx, cnt)
    return mx

def maxVowels(s: str, k: int) -> int:
    v = set('aeiou')
    cnt = 0
    r = 0
    while r < k:
        if s[r] in v:
            cnt += 1
        r += 1
    
    mx = cnt
    for i in range(r, len(s)):
        if s[i] in v:
            cnt += 1
        if s[i - k] in v:
            cnt -= 1
        mx = max(mx, cnt)
    return mx

#https://leetcode.com/problems/max-number-of-k-sum-pairs/?envType=study-plan-v2&envId=leetcode-75
def maxOperations(nums: list[int], k: int) -> int:
    nums = sorted(nums)
    cnt = 0
    l = 0
    r = len(nums) - 1
    while l < r:
        sm = nums[l] + nums[r]
        if sm == k:
            cnt += 1
            l += 1
            r -= 1
        elif sm > k:
            r -= 1
        else:
            l += 1
    return cnt

def maxOperations(nums: list[int], k: int) -> int:
    c = collections.Counter(nums)
    cnt = 0
    for v in c:
        rem = k - v
        if rem in c:
            if rem == v and c[v] > 1:
                cnt += c[v] // 2
                c[v] %= 2   
            elif rem != v:
                delta = min(c[rem], c[v])
                c[rem] -= delta
                c[v] -= delta
                cnt += delta      
    return cnt

#https://leetcode.com/problems/make-array-elements-equal-to-zero/?envType=daily-question&envId=2025-10-28
def countValidSelections(nums: list[int]) -> int:
    rsm = sum(nums)
    lsm = 0
    cnt = 0
    for i in range(len(nums)):
        if nums[i] > 0:
            lsm += nums[i]
            rsm -= nums[i]
        else:
            if lsm == rsm:
                cnt += 2
            elif abs(lsm - rsm) == 1:
                cnt += 1
    return cnt

#https://leetcode.com/problems/combination-sum-iii/?envType=study-plan-v2&envId=leetcode-75
def combinationSum3(k: int, n: int) -> list[list[int]]:
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    combs = []
    curr = []
    def _sub(i, rem):
        if len(curr) == k or i == len(nums):
            if len(curr) == k and rem == 0: #don't use sum() as it's O(n)
                combs.append(curr[:])
            return

        curr.append(nums[i])
        _sub(i + 1, rem - nums[i])
        curr.pop()
        _sub(i + 1, rem)
    _sub(0, n)
    return combs

#https://leetcode.com/problems/smallest-number-with-all-set-bits/?envType=daily-question&envId=2025-10-29
def smallestNumber(n: int) -> int:
    nb = bin(n)[2:]
    return int(nb.replace("0", "1"), 2)

def smallestNumber(n: int) -> int:
    p = floor(log2(n)) + 1
    return 2**p - 1 or 1

#https://leetcode.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/?envType=study-plan-v2&envId=leetcode-75
def minReorder(n: int, connections: list[list[int]]) -> int:
    d = defaultdict(list)
    for s, e in connections:
        d[s].append(e)
        d[e].append(-s)
    checked = [False] * n
    cnt = 0
    def _dfs(s):
        nonlocal cnt, d, checked
        checked[s] = True
        for e in d[s]:
            if not checked[abs(e)]:
                _dfs(abs(e))
                if e > 0: cnt += 1
    _dfs(0)
    return cnt

#https://leetcode.com/problems/minimum-number-of-increments-on-subarrays-to-form-a-target-array/?envType=daily-question&envId=2025-10-30
def minNumberOperations(target: list[int]) -> int:
    res = target[0]
    for i in range(1, len(target)):
        if target[i] > target[i - 1]:
            res += target[i] - target[i - 1]
    return res

#https://leetcode.com/problems/number-of-provinces/?envType=study-plan-v2&envId=leetcode-75
#this solution beats 5%😭😭😭
def findCircleNum(isConnected: list[list[int]]) -> int:
    l = len(isConnected)
    cnt = 0
    def _dfs(pi, ci):
        curr = isConnected[ci]
        if curr[pi] == 0: return 0

        curr[pi] = 0
        for j in range(l):
            if curr[j]: 
                _dfs(ci, j)
        return 1
            
    for i in range(l):
        cnt += _dfs(i, i)
    return cnt

def findCircleNum(isConnected: list[list[int]]) -> int:
    nv = set([i for i in range(len(isConnected))])
    v = set()
    cnt = 0
    while len(nv) > 0:
        curr = nv.pop()
        avr = {curr}
        cnt += 1
        while len(avr) > 0:
            cr = avr.pop()
            v.add(cr)
            routes = set([i for i in range(len(isConnected[cr])) if isConnected[cr][i]])
            routes -= v
            avr |= routes
        nv -= v
    return cnt

#https://leetcode.com/problems/keys-and-rooms/?envType=study-plan-v2&envId=leetcode-75
def canVisitAllRooms(rooms: list[list[int]]) -> bool:
    visitable = [True] + [False] * (len(rooms) - 1)
    def _dfs(r):
        if not visitable[r]: 
            return
        
        for k in rooms[r]:
            if visitable[k]: continue
            visitable[k] = True
            _dfs(k)
    _dfs(0)
    return all(visitable)

def canVisitAllRooms(rooms: list[list[int]]) -> bool:
    visited = [True] + [False] * (len(rooms) - 1)
    queue = collections.deque([0])
    while queue:
        r = queue.popleft()
        for k in rooms[r]:
            if not visited[k]:
                visited[k] = True
                queue.append(k)
    return all(visited)

#https://leetcode.com/problems/rotting-oranges/?envType=study-plan-v2&envId=leetcode-75
def orangesRotting(grid: list[list[int]]) -> int:
    w, h = len(grid[0]), len(grid)
    q = collections.deque()
    k = 0
    for y in range(h):
        for x in range(w):
            if grid[y][x] == 2: q.append((x, y))
            elif grid[y][x] == 1: k += 1
    if not q: 
        if not k: return 0
        else: return -1

    t = -1 
    while q:
        for _ in range(len(q)):
            x, y = q.popleft()
            for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w) or not (0 <= ny < h) or grid[ny][nx] != 1: continue
                q.append((nx, ny))
                grid[ny][nx] = 2
                k -= 1
        t += 1
    return t if k == 0 else -1

#inefficient af, but added just in case
def orangesRotting(grid: list[list[int]]) -> int:
    w, h = len(grid[0]), len(grid)
    def _bfs(x, y):
        q = collections.deque([(x, y)])
        chk = set([(x, y)])
        d = 0
        while q:
            for _ in range(len(q)):
                x, y = q.popleft()
                if grid[y][x] == 2: return d
                for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < w) or not (0 <= ny < h) or (nx, ny) in chk or grid[ny][nx] == 0: continue
                    q.append((nx, ny))
                    chk.add((nx, ny))
            d += 1
        return None

    mn = 0
    for y in range(h):
        for x in range(w):
            if grid[y][x] == 1: 
                mnt = _bfs(x, y)
                if mnt == None: return -1
                else: mn = max(mn, mnt)
    return mn

#https://leetcode.com/problems/nearest-exit-from-entrance-in-maze/?envType=study-plan-v2&envId=leetcode-75
#beats 10.51%😭😭😭
def nearestExit(maze: list[list[str]], entrance: list[int]) -> int:
    m = len(maze)
    n = len(maze[0])
    entrance = [entrance[1], entrance[0]]
    queue = collections.deque([(*entrance, 0)])
    def _get(x, y, steps):
        for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            nx = x + dx
            ny = y + dy
            if not (0 <= ny < m) or not (0 <= nx < n) or maze[ny][nx] == "+" or maze[ny][nx] == "0": continue
            else:
                maze[ny][nx] = "0"
                queue.append((nx, ny, steps + 1))
    
    mn = float("inf")
    while queue:
        x, y, steps = queue.popleft()
        if ([x, y] != entrance) and (x == 0 or x == n - 1 or y == 0 or y == m - 1): 
            mn = min(mn, steps)
        else:
            _get(x, y, steps)
    if not queue and mn == float("inf"): return -1
    else: return mn

def nearestExit(maze: list[list[str]], entrance: list[int]) -> int:
    m = len(maze)
    n = len(maze[0])
    queue = collections.deque([(entrance[1], entrance[0], 0)])
    def _get(x, y, steps):
        for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            nx = x + dx
            ny = y + dy
            if not (0 <= ny < m) or not (0 <= nx < n) or maze[ny][nx] == "+": continue
            else:
                maze[ny][nx] = "+"
                queue.append((nx, ny, steps + 1))
    
    while queue:
        x, y, steps = queue.popleft()
        if ([y, x] != entrance) and (x == 0 or x == n - 1 or y == 0 or y == m - 1): 
            return steps #the first route that reaches exit already takes min steps, so no need for min()
        else:
            _get(x, y, steps) 
    return -1

#https://leetcode.com/problems/maximum-twin-sum-of-a-linked-list/?envType=study-plan-v2&envId=leetcode-75
def pairSum(head: ListNode) -> int:
    sums = []
    while head:
        sums.append(head.val)
        head = head.next
    l = 0
    r = len(sums) - 1
    mx = 0
    while l < r:
        mx = max(sums[l] + sums[r], mx)
        l += 1
        r -= 1
    return mx

def pairSum(head: ListNode) -> int:
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    nxt = None
    while slow:
        temp = slow.next
        slow.next = nxt
        nxt = slow
        slow = temp
    
    mx = 0
    while nxt and head:
        mx = max(mx, nxt.val + head.val)
        head = head.next
        nxt = nxt.next
    return mx

#https://leetcode.com/problems/odd-even-linked-list/?envType=study-plan-v2&envId=leetcode-75
def oddEvenList(head: ListNode) -> ListNode:
    odd = odd_d = ListNode()
    even = even_d = ListNode()

    n = 0
    while head:
        temp = head
        head = head.next
        temp.next = None
        if n % 2 == 0:
            even.next = temp
            even = even.next
        else:
            odd.next = temp
            odd = odd.next
        n += 1
    even.next = odd_d.next
    return even_d.next

#https://leetcode.com/problems/delete-the-middle-node-of-a-linked-list/?envType=study-plan-v2&envId=leetcode-75
def deleteMiddle(head: ListNode) -> ListNode:
    prev = slow = fast = head
    while fast and fast.next:
        prev = slow
        slow = slow.next
        fast = fast.next.next
    if prev == slow == fast:
        return 
    else:
        prev.next = slow.next
        return head

#https://leetcode.com/problems/power-grid-maintenance/?envType=daily-question&envId=2025-11-06  
def processQueries(c: int, connections: list[list[int]], queries: list[list[int]]) -> list[int]:
    parent = list(range(c + 1))
    def find(x):
        if parent[x] == x: return x
        else: 
            parent[x] = find(parent[x])
            return parent[x]

    # union connected stations
    for a, b in connections:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pb] = pa

    mn = [0] * (c + 1)
    nxt = [0] * (c + 1)
    last = {}
    for i in range(1, c + 1):
        p = find(i)
        if mn[p] == 0:
            mn[p] = i
        else:
            nxt[last[p]] = i
        last[p] = i
    
    active = [True] * (c + 1)
    res = []
    for op, s in queries:
        if op == 1:
            if active[s]:
                res.append(s)
            else:
                p = find(s)
                res.append(mn[p] if mn[p] else -1)
        elif op == 2 and active[s]:
            active[s] = False
            p = find(s)
            if mn[p] == s:
                n = nxt[s]
                while n and not active[n]:
                    n = nxt[n]
                mn[p] = n if n else 0
    return res

#https://leetcode.com/problems/search-in-a-binary-search-tree/?envType=study-plan-v2&envId=leetcode-75
def searchBST(root: TreeNode, val: int) -> TreeNode:
    while root:
        if root.val == val:
            return root
        elif root.val > val:
            root = root.left
        else:
            root = root.right
    return None

#https://leetcode.com/problems/leaf-similar-trees/?envType=study-plan-v2&envId=leetcode-75
def leafSimilar(root1: TreeNode, root2: TreeNode) -> bool:
    def _dfs(root):
        if not root: 
            return []
        if not root.left and not root.right:
            return [root.val]
        
        return _dfs(root.left) + _dfs(root.right)
    return _dfs(root1) == _dfs(root2)

#https://leetcode.com/problems/number-of-recent-calls/?envType=study-plan-v2&envId=leetcode-75
class RecentCounter:
    def __init__(self):
        self.stack = []

    def ping(self, t: int) -> int:
        self.stack.append(t)
        i = len(self.stack) - 1
        k = 0
        while i >= 0 and t - 3000 <= self.stack[i] <= t:
            k += 1
            i -= 1
        return k
    
class RecentCounter:
    def __init__(self):
        self.stack = []

    def ping(self, t: int) -> int:
        self.stack.append(t)
        while self.stack[0] < t - 3000:
            self.stack.pop(0)
        return len(self.stack)

#https://leetcode.com/problems/maximize-the-minimum-powered-city/editorial/?envType=daily-question&envId=2025-11-07   
def maxPower(stations: list[int], r: int, k: int):
    n = len(stations)
    df = [0] * n
    # Build initial power difference array
    for i in range(len(stations)):
        df[max(0, i - r)] += stations[i]
        if i + r + 1 < n:
            df[i + r + 1] -= stations[i]

    act_df = df.copy()
    for i in range(1, len(stations)):
        act_df[i] += act_df[i - 1]

    def _chk(mn):
        diff = df.copy()
        curr = cnt = 0
        for i in range(n):
            curr += diff[i]
            if curr < mn:
                delta = mn - curr
                cnt += delta
                if cnt > k:
                    return False
                curr = mn
                if i + 2 * r + 1 < n:
                    diff[i + 2 * r + 1] -= delta
        return True
    
    low, high = min(act_df), 2 * 10**10
    while low < high:
        mid = (low + high + 1) >> 1
        if _chk(mid):
            low = mid
        else:
            high = mid - 1
    return low    

#https://leetcode.com/problems/minimum-one-bit-operations-to-make-integers-zero/?envType=daily-question&envId=2025-11-08
def minimumOneBitOperations(n: int) -> int:
    if n == 0: return 0
    nb = bin(n)[2:]
    l = len(nb)
    ops = 0
    sign = 1
    for i in range(l):
        if nb[i] == "1":
            ops += sign * (2**(l - i) - 1)
            sign *= -1
    return ops

#https://leetcode.com/problems/count-operations-to-obtain-zero/?envType=daily-question&envId=2025-11-09
def countOperations(num1: int, num2: int) -> int:
    ops = 0
    while num1 != 0 and num2 != 0:
        if num1 >= num2:
            num1 = num1 - num2
        else:
            num2 = num2 - num1
        ops += 1
    return ops

#https://leetcode.com/problems/redundant-connection/?envType=problem-list-v2&envId=union-find
def findRedundantConnection(edges: list[list[int]]) -> list[int]:
    parent = [i for i in range(len(edges) + 1)]
    def _find(x):
        if parent[x] == x: return x
        else:
            return _find(parent[x])
    
    for a, b in edges:
        pa, pb = _find(a), _find(b)
        if pa != pb:
            parent[pb] = parent[pa]
        else:
            return [a, b]

def findRedundantConnection(edges: list[list[int]]) -> list[int]:
    adj = defaultdict(list)     
    
    def _dfs(node, parent):
        if node in v: return False

        v.add(node)
        for snode in adj[node]:
            if snode == parent: continue
            if not _dfs(snode, node): return False
        return True

    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        v = set()
        if not _dfs(a, b): return [a, b]

def findRedundantConnection(edges: list[list[int]]) -> list[int]:
    adj = defaultdict(list)  
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)   
    
    v = set()
    cycle, cst = set(), None

    def _dfs(node, parent):
        nonlocal cst
        if node in v:
            cst = node
            cycle.add(node)
            return True

        v.add(node)
        for snode in adj[node]:
            if snode == parent: continue
            if _dfs(snode, node):
                if cst == node: cst = None #stop adding nodes to cycle after reaching cycle start
                elif cst != None: cycle.add(node)
                return True
        return False

    _dfs(1, -1)
    for a, b in reversed(edges): #last in the input
        if a in cycle and b in cycle: return [a, b]

#https://leetcode.com/problems/minimum-operations-to-convert-all-elements-to-zero/?envType=daily-question&envId=2025-11-10
def minOperations(nums: list[int]) -> int:
    stack = collections.deque()
    ops = 0
    for n in nums:
        while stack and n < stack[0]:
            stack.popleft()
        if n == 0: continue
        if not stack or n > stack[0]:
            stack.appendleft(n)
            ops += 1
    return ops        

#https://leetcode.com/problems/ones-and-zeroes/?envType=daily-question&envId=2025-11-11
def findMaxForm(strs: list[str], m: int, n: int) -> int:
    d = {(0, 0): 0}
    for s in strs:
        o, z = s.count("1"), s.count("0")
        nd = {}
        for (po, pz), cnt in d.items():
            co, cz = o + po, z + pz
            if co <= n and cz <= m:
                dcnt = d.get((co, cz), False)
                if not dcnt or dcnt < cnt + 1:
                    nd[(co, cz)] = cnt + 1               
        d.update(nd)
    return max(d.values())

#https://leetcode.com/problems/minimum-number-of-operations-to-make-all-array-elements-equal-to-1/?envType=daily-question&envId=2025-11-12
def minOperations(nums: list[int]) -> int:
    n = len(nums)
    ones = nums.count(1)
    if ones: return n - ones

    res = inf
    for i in range(n):
        for j in range(i + 1, n):
            nums[i] = gcd(nums[i], nums[j])
            if nums[i] == 1:
                res = min(res, j - i)
    if res == inf: return -1
    return res + n - 1

#https://leetcode.com/problems/maximum-number-of-operations-to-move-ones-to-the-end/?envType=daily-question&envId=2025-11-13
def maxOperations(s: str) -> int:
    ops = ones = 0
    i = 0
    while i < len(s):
        if s[i] == "1":
            ones += 1
            i += 1
        elif s[i] == "0":
            ops += ones
            while i < len(s) and s[i] == "0":
                i += 1    
    return ops

def maxOperations(s: str) -> int:
    ops = ones = 0
    for i in range(len(s) - 1):
        if s[i] == "1":
            ones += 1
            if s[i + 1] == "0":
                ops += ones
    return ops

#https://leetcode.com/problems/increment-submatrices-by-one/?envType=daily-question&envId=2025-11-14
def rangeAddQueries(n: int, queries: list[list[int]]) -> list[list[int]]:
    diff = [[0] * (n) for _ in range(n)]
    for q in queries:
        y0, x0, y1, x1 = q
        diff[y0][x0] += 1
        if x1 + 1 < n: diff[y0][x1 + 1] -= 1
        if y1 + 1 < n: diff[y1 + 1][x0] -= 1
        if y1 + 1 < n and x1 + 1 < n: diff[y1 + 1][x1 + 1] += 1
        
    for y in range(n):
        for x in range(n):
            u = diff[y - 1][x] if y > 0 else 0
            l = diff[y][x - 1] if x > 0 else 0
            d = diff[y - 1][x - 1] if y > 0 and x > 0 else 0
            diff[y][x] += u + l - d
    return diff

#https://leetcode.com/problems/count-the-number-of-substrings-with-dominant-ones/?envType=daily-question&envId=2025-11-15
#didn't understand a shit, "medium" problem btw
def numberOfSubstrings(s: str) -> int:
    n = len(s)
    pre = [-1] * (n + 1)
    for i in range(n):
        if i == 0 or s[i - 1] == "0":
            pre[i + 1] = i
        else:
            pre[i + 1] = pre[i]

    res = 0
    for i in range(1, n + 1):
        z = s[i - 1] == "0"
        j = i
        while j > 0 and z**2 < n:
            o = (i - pre[j]) - z
            if z**2 <= o:
                res += min(j - pre[j], o - z**2 + 1)
            j = pre[j]
            z += 1
    return res

#https://leetcode.com/problems/number-of-substrings-with-only-1s/?envType=daily-question&envId=2025-11-16
def numSub(s: str) -> int:
    counts = [len(i) for i in s.split("0")]
    res = 0
    for c in counts:    
        res += (c + 1) * c // 2
    return res % (10**9 + 7)

#https://leetcode.com/problems/check-if-all-1s-are-at-least-length-k-places-away/?envType=daily-question&envId=2025-11-17
def kLengthApart(nums: list[int], k: int) -> bool:
    delta = 0
    j = 0
    n = len(nums)
    while j < n and nums[j] != 1:
        j += 1
    for i in range(j + 1, n):   
        if nums[i] == 0:
            delta += 1
        else:
            if delta >= k: 
                delta = 0
            else:
                return False
    return True

#https://leetcode.com/problems/redundant-connection-ii/?envType=problem-list-v2&envId=union-find
def findRedundantDirectedConnection(edges: list[list[int]]) -> list[int]:
    n1 = len(edges) + 1
    parent = [i for i in range(n1)]
    p_e = [-1 for _ in range(n1)]
    def _find(x):
        if parent[x] == x: return x
        else: 
            parent[x] = _find(parent[x])
            return parent[x]

    first = second = last = -1
    for i in range(len(edges)):
        a, b = edges[i][0], edges[i][1]
        if p_e[b] != -1:
            first = p_e[b] 
            second = i
        else:
            p_e[b] = i
            pa = _find(a)
            if pa == b: last = i
            else: parent[b] = pa
    
    if last == -1: return edges[second]
    elif second == -1: return edges[last]
    else: return edges[first]

#https://leetcode.com/problems/1-bit-and-2-bit-characters/?envType=daily-question&envId=2025-11-18
def isOneBitCharacter(bits: list[int]) -> bool:
    n = len(bits)
    d = 0
    p = None
    i = 0
    while i < len(bits):
        p = (bits[i], bits[i + 1] if i < n - 1 else None)
        if p[0] == 0: 
            d += 1
            i += 1
        else:
            d += 2
            i += 2
    return p[0] == 0 and d == n

def isOneBitCharacter(bits: list[int]) -> bool:
    n = len(bits)
    i = 0
    while i < n - 1:
        if bits[i] == 1:
            i += 2
        else:
            i += 1
    return i == n - 1

#https://leetcode.com/problems/keep-multiplying-found-values-by-two/submissions/1834249651/?envType=daily-question&envId=2025-11-19
def findFinalValue(nums: list[int], original: int) -> int:
    res = original
    seen = set(nums)
    while res in seen:
        res <<= 1
    return res

#https://leetcode.com/problems/set-intersection-size-at-least-two/
def intersectionSizeTwo(intervals: list[list[int]]) -> int:
    intervals = sorted(intervals, key=lambda x: (x[1], -x[0]))
    window = []
    n = 0
    for start, end in intervals:
        if not window or start > window[1]:
            n += 2
            window = [end - 1, end]
        elif start > window[0]:
            n += 1
            window = [window[1], end]
    return n

#https://leetcode.com/problems/unique-length-3-palindromic-subsequences/?envType=daily-question&envId=2025-11-21
def countPalindromicSubsequence(s: str) -> int:
    s = list(s)
    pos = {}
    for i in range(len(s)):
        char = s[i]
        if not pos.get(char, False):
            pos[char] = [i, i]
        else:
            pos[char][1] = i
    
    n = 0
    for c in s:
        l, r = pos[c]
        if l == r: continue
        else:       
            n += len(set(s[l+1:r]))
            pos[c][0] = pos[c][1]
    return n

def countPalindromicSubsequence(s: str) -> int:
    if len(s) <= 2: return 0
    
    chars = set(s)
    n = 0
    for c in chars:
        l, r = s.find(c), s.rfind(c)
        if l == r: continue
        else:
            n += len(set(s[l+1:r]))
    return n

#https://leetcode.com/problems/find-minimum-operations-to-make-all-elements-divisible-by-three/?envType=daily-question&envId=2025-11-22
def minimumOperations(nums: list[int]) -> int:
    ops = 0
    for n in nums:
        rem = n % 3
        ops += min(rem, 3 - rem)
    return ops

#https://leetcode.com/problems/max-area-of-island/?envType=problem-list-v2&envId=union-find
def maxAreaOfIsland(grid: list[list[int]]) -> int:
    m = len(grid)
    n = len(grid[0])
    def _get(x, y):
        if not (0 <= x < n) or not (0 <= y < m) or not grid[y][x]:
            return 0
        
        grid[y][x] = 0
        return 1 + _get(x + 1, y) + _get(x, y + 1) + _get(x - 1, y) + _get(x, y - 1)
    
    mx = 0
    for y in range(m):
        for x in range(n):
            if grid[y][x]:
                mx = max(_get(x, y), mx)
    return mx

def maxAreaOfIsland(grid: list[list[int]]) -> int:
    w, h = len(grid[0]), len(grid)
    
    mx = 0
    for y in range(h):
        for x in range(w):
            if grid[y][x] == 0: continue

            k = 1
            q = collections.deque([(x, y)])
            grid[y][x] = 0
            while q:
                x, y = q.popleft()
                for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < w) or not (0 <= ny < h) or grid[ny][nx] == 0: continue
                    k += 1
                    q.append((nx, ny))
                    grid[ny][nx] = 0
            mx = max(mx, k)
    return mx

def maxAreaOfIsland(grid: list[list[int]]) -> int:
    w, h = len(grid[0]), len(grid)
    def _idx(x, y): return y * w + x

    p = list(range(w * h))
    p_pwr = list([1 for _ in range(w * h)])
    def _find(x):
        if p[x] == x: return x
        else:
            p[x] = _find(p[x])
            return p[x]
    
    for y in range(h):
        for x in range(w):
            if grid[y][x] == 0: 
                p_pwr[_idx(x, y)] = 0
                continue

            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w) or not (0 <= ny < h) or grid[ny][nx] == 0: continue
                pa, pb = _find(_idx(x, y)), _find(_idx(nx, ny))
                if pa != pb: 
                    if p_pwr[pa] >= p_pwr[pb]:
                        p[pb] = pa
                        p_pwr[pa] += p_pwr[pb]
                        p_pwr[pb] = 0
                    else:
                        p[pa] = pb
                        p_pwr[pb] += p_pwr[pa]
                        p_pwr[pa] = 0
    return max(p_pwr)

#https://leetcode.com/problems/greatest-sum-divisible-by-three/?envType=daily-question&envId=2025-11-23
def maxSumDivThree(nums: list[int]) -> int:
    nums = sorted(nums)
    sm = 0
    r1, r2 = [], []
    for n in nums:
        sm += n
        rem = n % 3
        if rem == 1 and len(r1) < 2: r1.append(n)
        elif rem == 2 and len(r2) < 2: r2.append(n)

    smrem = sm % 3
    if smrem == 0: return sm
    elif smrem == 1:
        s1 = r1[0] if r1 else float("inf")
        s2 = sum(r2) if len(r2) == 2 else float("inf")
    else:
        s1 = sum(r1) if len(r1) == 2 else float("inf")
        s2 = r2[0] if r2 else float("inf")
        
    sm = sm - min(s1, s2)
    return sm if sm != float('inf') else 0

def maxSumDivThree(nums: list[int]) -> int:
    n = len(nums)
    @cache 
    def _find(i, rem):
        if i < 0: return 0 if rem == 0 else -(1<<32)
        x = nums[i]
        prev_rem = (rem - x) % 3
        #if prev_rem < 0: prev_rem += 3
        return max(x + _find(i - 1, prev_rem), _find(i - 1, rem))
    return max(0, _find(n - 1, 0))

#https://leetcode.com/problems/binary-prefix-divisible-by-5/?envType=daily-question&envId=2025-11-24
def prefixesDivBy5(nums: list[int]) -> list[bool]:
    div = [False] * len(nums)
    curr = 0
    for i in range(len(nums)):
        curr <<= 1
        curr += nums[i]
        if curr % 5 == 0: div[i] = True
    return div

def prefixesDivBy5(nums: list[int]) -> list[bool]:
    div = []
    rem = 0
    for n in nums:
        rem = (rem * 2 + n) % 5
        div.append(rem == 0)
    return div

#https://leetcode.com/problems/accounts-merge/?source=submission-noac
#DID IT MYSELF GOOOOAL
def accountsMerge(accounts: list[list[str]]) -> list[list[str]]:
    n = len(accounts)
    parent = [i for i in range(n)]
    def _find(x):
        if parent[x] == x: return x
        else: return _find(parent[x])
        
    rel = {}
    checked = set()
    for i in range(n):
        acc = accounts[i]
        emails = set(acc[1:])
        inter = checked & emails
        #so, the main idea is that if we find intersection we merge this acc and a remaining accs in inter with the last acc in inter
        if inter:
            mp = _find(rel[inter.pop()]) #main parent (3rd group to merge with)
            ip = _find(i)
            parent[ip] = mp
            for e in inter:
                ep = _find(rel[e])
                if ep != mp:
                    parent[ep] = mp
        for e in emails:
            if not rel.get(e, False):
                rel[e] = i
        checked |= emails
    
    #merge itself
    for i in range(n):
        pi = _find(i)
        if pi != i:
            accounts[pi].extend(accounts[i][1:])
            accounts[i][0] = None
    
    #deleting empty accs
    i = 0
    while i < n:
        if accounts[i][0] == None: 
            del accounts[i]
        else:
            accounts[i] = [accounts[i][0], *sorted(set(accounts[i][1:]))]
            i += 1
        n = len(accounts)
    return accounts


def accountsMerge(accounts: list[list[str]]) -> list[list[str]]:
    n = len(accounts)
    parent = [i for i in range(n)]
    def _find(x):
        if parent[x] == x: return x
        else: 
            parent[x] = _find(parent[x])
            return parent[x]
        
    #optimized solution without sets
    rel = {}
    for i, (_, *emails) in enumerate(accounts):
        for e in emails:
            if e in rel:
                pi, pg = _find(i), _find(rel[e]) #instead of selecting a random parent, we assign to previous one
                parent[pi] = pg
            rel[e] = i #than reassign this parent to be the next's previous (Xd)
    
    res = defaultdict(list)
    for e, p in rel.items():
        res[_find(p)].append(e)
    return [[accounts[i][0]] + sorted(emails) for i, emails in res.items()]

#https://leetcode.com/problems/smallest-integer-divisible-by-k/?envType=daily-question&envId=2025-11-25
def smallestRepunitDivByK(k: int) -> int:
    if k % 2 == 0 or k % 5 == 0: return -1
    rems = set()
    n = 1
    rem = 0
    while True:
        rem = (rem * 10 + 1) % k
        if rem in rems:
            return -1
        elif rem == 0:   
            return n
        n += 1

#https://leetcode.com/problems/paths-in-matrix-whose-sum-is-divisible-by-k/?envType=daily-question&envId=2025-11-26
def numberOfPaths(grid: list[list[int]], k: int) -> int:
    MOD = (10**9 + 7)
    m1, n1 = len(grid) + 1, len(grid[0]) + 1
    
    d = [[[0] * k for _ in range(n1)] for _ in range(m1)]
    for y in range(1, m1):
        for x in range(1, n1):
            if y == 1 and x == 1: 
                d[y][x][grid[0][0] % k] = 1
                continue
            v = grid[y - 1][x - 1] % k
            for rem in range(k):
                prev = (rem - v) % k #prev + v =modk= rem => prev =modk= rem - v
                d[y][x][rem] = (d[y - 1][x][prev] + d[y][x - 1][prev]) % MOD
    return d[m1 - 1][n1 - 1][0]

#https://leetcode.com/problems/maximum-subarray-sum-with-length-divisible-by-k/?envType=daily-question&envId=2025-11-27
def maxSubarraySum(nums: list[int], k: int) -> int:
    mn_sm = [float('inf')] * (k - 1) + [0] #stores min sum of previous elements before k (i % k != 0, "cycle")
    mx = float('-inf')
    curr = 0
    for i in range(len(nums)):
        curr += nums[i]
        pos = i % k #if pos == 0, new "cycle" begins
        mx = max(mx, curr - mn_sm[pos])
        mn_sm[pos] = min(mn_sm[pos], curr)
    return mx

#https://leetcode.com/problems/maximum-number-of-k-divisible-components/?envType=daily-question&envId=2025-11-28
def maxKDivisibleComponents(n: int, edges: list[list[int]], values: list[int], k: int) -> int:
    rel = defaultdict(list)
    for a, b in edges:
        rel[a].append(b)
        rel[b].append(a)   

    chunks = 0
    def _dfs(e, pe):
        nonlocal chunks 
        rem = values[e] % k
        for ce in rel[e]:
            if ce != pe:
                rem = (rem + _dfs(ce, e)) % k
        if rem == 0:
            chunks += 1
        return rem #if chunk is divisible by k, it returns 0, so it doesn't affect former parent 
    _dfs(0, -1)
    return chunks

#https://leetcode.com/problems/minimum-operations-to-make-array-sum-divisible-by-k/?envType=daily-question&envId=2025-11-29
def minOperations(nums: list[int], k: int) -> int:
    sm = sum(nums)
    return sm % k

#https://leetcode.com/problems/is-graph-bipartite/?envType=problem-list-v2&envId=union-find
def isBipartite(graph: list[list[int]]) -> bool:
    color = [None] * len(graph)
    def _dfs(e):
        for ce in graph[e]:
            if color[ce] == color[e]: return False
            elif color[ce] is None: #if node is colored and color[ce] != color[e] it will skip the node
                color[ce] = 1 - color[e]
                if not _dfs(ce): return False
        return True

    for e in range(len(graph)):
        if color[e] is None:
            color[e] = 0
            if not _dfs(e): return False
    return True

def isBipartite(graph: list[list[int]]) -> bool:
    color = [None] * len(graph)
    def _bfs(e):
        q = collections.deque([e])
        while q:
            e = q.popleft()
            for ce in graph[e]:
                if color[ce] == color[e]: return False
                elif color[ce] == None: 
                    color[ce] = 1 - color[e]
                    q.append(ce)
        return True

    for e in range(len(graph)):
        if color[e] == None:
            color[e] = 0
            if not _bfs(e): return False
    return True

#https://leetcode.com/problems/make-sum-divisible-by-p/?envType=daily-question&envId=2025-11-30
def minSubarray(nums: list[int], p: int) -> int:
    #(curri - currj) % p = rem => currj =- (curri - rem) % p => something something => req = (curr - rem + p) % p
    rem = sum(nums) % p
    if rem == 0: return 0

    mod = {0: -1}
    mn = float("inf")
    curr = 0
    for i in range(len(nums)):
        curr += nums[i]
        req = (curr - rem + p) % p #missing required remainder
        if req in mod:
            mn = min(i - mod[req], mn)
        mod[curr % p] = i

    if mn == float("inf") or mn == len(nums): return -1
    else: return mn

#https://leetcode.com/problems/maximum-running-time-of-n-computers/?envType=daily-question&envId=2025-12-01
def maxRunTime(n: int, batteries: list[int]) -> int:
    batteries = sorted(batteries)
    available = sum(batteries[:-n])
    running = batteries[-n:]
    for i in range(n - 1):
        if available // (i + 1) < running[i + 1] - running[i]:
            return running[i] + available // (i + 1)
        else:
            available -= (i + 1) * (running[i + 1] - running[i])
    return running[-1] + available // n


#https://leetcode.com/problems/count-number-of-trapezoids-i/?envType=daily-question&envId=2025-12-02
#beats 18% but i did it myself Xd
def countTrapezoids(points: list[list[int]]) -> int:
    lines = defaultdict(int)
    for _, y in points:
        lines[y] += 1
    
    l_cnt = []
    sm = 0
    for y in lines:
        if lines[y]:
            v = comb(lines[y], 2) #number of ways to select 2 points to form a lines
            sm += v
            l_cnt.append(v)
            
    total = 0
    for cnt in l_cnt:
        sm -= cnt
        total += cnt * sm
    return total % (10**9 + 7) 

#https://leetcode.com/problems/count-number-of-trapezoids-ii/?envType=daily-question&envId=2025-12-03
#Я очень устал, босс. Я очень устал...
def countTrapezoids(points: list[list[int]]) -> int:
    n = len(points)
    t = defaultdict(lambda: defaultdict(int))
    v = defaultdict(lambda: defaultdict(int))
    def _count(groups):
        res = 0
        for group in groups.values():
            total = sum(group.values())
            #it only works if there are 2+ lines within the group (requirement for trapezoid)
            for v in group.values():
                total -= v
                res += v * total
        return res #total number of figures i guess

    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1

            if dx < 0 or (dx == 0 and dy < 0): #canonical sign thingy
                dx = -dx
                dy = -dy

            g = gcd(dx, abs(dy))
            #reduced direction (rx, ry)
            rx = dx // g
            ry = dy // g
            line = rx * y1 - ry * x1 #constant for a line

            key1 = (rx << 12) | (ry + 2000) #hash instead of (rx, ry) type keys
            key2 = (dx << 12) | (dy + 2000)

            #basically, instead of a previously long comparison by endpoints, it compares by aforementioned line constant
            t[key1][line] += 1 #how many groups have this slope
            v[key2][line] += 1 #how many groups have this vector

    return _count(t) - _count(v) // 2

#https://leetcode.com/problems/count-collisions-on-a-road/?envType=daily-question&envId=2025-12-04
def countCollisions(directions: str) -> int:
    directions = list(directions)
    stack = collections.deque()
    res = 0
    for d in directions:
        if not stack:
            stack.append([d, 1])
        else:
            pd, pcnt = stack.pop()
            if (d == "S" and pd == "R") or (pd == "S" and d == "L"):
                if pd == "R": res += pcnt
                else: res += 1
                
                if stack and stack[-1][0] == "R":
                    _, pcnt = stack.pop()
                    res += pcnt
                stack.append(["S", 1])
            elif pd == "R" and d == "L":
                res += 1 + pcnt
                if stack and stack[-1][0] == "R":
                    _, pcnt = stack.pop()
                    res += pcnt 
                stack.append(["S", 1])
            else:
                if pd == d: stack.append([pd, pcnt + 1])
                else:
                    stack.append([pd, 1])
                    stack.append([d, 1])
    return res
    
def countCollisions(directions: str) -> int:
    directions = directions.lstrip('L').rstrip('R')
    return len(directions) - directions.count('S')

#https://leetcode.com/problems/count-partitions-with-even-sum-difference/?envType=daily-question&envId=2025-12-05
def countPartitions(nums: list[int]) -> int:
    l, r = 0, sum(nums)
    k = 0 
    for n in nums[:-1]:
        l += n
        r -= n
        if abs(l - r) % 2 == 0: k += 1
    return k

def countPartitions(nums: list[int]) -> int:
    if sum(nums) % 2: return 0
    else: return len(nums) - 1

#https://leetcode.com/problems/count-partitions-with-max-min-difference-at-most-k/solutions/7394846/all-language-solution-c-java-python-rust-tv0k/?envType=daily-question&envId=2025-12-06
#demotivated
def countPartitions(nums: list[int], k: int) -> int:
    n = len(nums)
    MOD = 10**9 + 7
    
    mx, mn = collections.deque(), collections.deque()
    dp = [0] * (n + 1)
    dp[0] = 1
    s = 0
    l = 0
    
    for r in range(n):
        while mx and nums[mx[-1]] <= nums[r]:
            mx.pop()
        while mn and nums[mn[-1]] >= nums[r]:
            mn.pop()
        mx.append(r)
        mn.append(r)
        
        while nums[mx[0]] - nums[mn[0]] > k:
            if mx[0] == l:
                mx.popleft()
            if mn[0] == l:
                mn.popleft()
            s = (s - dp[l]) % MOD
            l += 1
        
        s = (s + dp[r]) % MOD
        dp[r + 1] = s
    
    return dp[n]

#https://leetcode.com/problems/count-odd-numbers-in-an-interval-range/?envType=daily-question&envId=2025-12-07
def countOdds(low: int, high: int) -> int:
    l = high - low
    if low % 2 != 0 or high % 2 != 0: return l // 2 + 1
    else: return l // 2 

#https://leetcode.com/problems/count-square-sum-triples/?envType=daily-question&envId=2025-12-08
def countTriples(n: int) -> int:
    k = 0
    cs = set([i**2 for i in range(1, n + 1)])
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if a**2 + b**2 in cs: k += 2
    return k

#https://leetcode.com/problems/count-special-triplets/?envType=daily-question&envId=2025-12-09
def specialTriplets(nums: list[int]) -> int:
    next_freq = collections.Counter(nums)
    prev_freq = defaultdict(int)

    k = 0
    for i in range(len(nums)):
        n = nums[i]
        next_freq[n] -= 1
        k += prev_freq[n * 2] * next_freq[n * 2]    
        prev_freq[n] += 1
    
    return k % (10**9 + 7)

def specialTriplets(nums: list[int]) -> int:
    idx = defaultdict(list)
    for i, n in enumerate(nums):
        idx[n].append(i)

    k = 0
    for i in range(len(nums)):
        arr = idx[nums[i]*2]
        l = bisect.bisect_left(arr, i) #i is not included, so left = i - 0 
        r = len(arr) - bisect.bisect_left(arr, i + 1) #i + 1 is included and r = n + 1 - i, so everything's alright
        k += l * r
    return k % (10**9 + 7)

#https://leetcode.com/problems/count-the-number-of-computer-unlocking-permutations/?envType=daily-question&envId=2025-12-10
def countPermutations(complexity: list[int]) -> int:
    for n in complexity[1:]:
        if n <= complexity[0]: return 0
    return factorial(len(complexity) - 1) % (10**9 + 7)

#https://leetcode.com/problems/count-covered-buildings/?envType=daily-question&envId=2025-12-11
#beats 8.41%😭😭😭
def countCoveredBuildings(n: int, buildings: list[list[int]]) -> int:
    bx = sorted(buildings, key=lambda x: x[0])
    by = sorted(buildings, key=lambda x: x[1])
    xc = defaultdict(list)
    yc = defaultdict(list)
    for x, y in by: xc[x].append((x, y)) 
    for x, y in bx: yc[y].append((x, y))

    k = 0
    for x in xc:
        points = xc[x][1:-1]
        for x, y in points:
            if 0 < yc[y].index((x, y)) < len(yc[y]) - 1: k += 1
    return k

def countCoveredBuildings(n: int, buildings: list[list[int]]) -> int:
    xmn, xmx = [n + 1] * (n + 1), [0] * (n + 1)
    ymn, ymx = [n + 1] * (n + 1), [0] * (n + 1)
    for x, y in buildings:
        xmn[x], xmx[x] = min(xmn[x], y), max(xmx[x], y) #max and min y in xth col
        ymn[y], ymx[y] = min(ymn[y], x), max(ymx[y], x) #max and min x in yth row

    k = 0
    for x, y in buildings:
        c1 = xmn[x] < y < xmx[x]
        c2 = ymn[y] < x < ymx[y]
        if c1 and c2: k += 1
    return k

#https://leetcode.com/problems/count-mentions-per-user/?envType=daily-question&envId=2025-12-12
def countMentions(numberOfUsers: int, events: list[list[str]]) -> list[int]:
    evs = defaultdict(list)
    timestamps = set()
    for e in events:
        t = int(e[1])
        timestamps.add(t)
        evs[t].append((e[0], e[2]))
        evs[t] = sorted(evs[t], key=lambda x: x[0], reverse=True)

    seen = [float("-inf")] * numberOfUsers
    mentions = [0] * numberOfUsers
    for t in sorted(timestamps):
        for e in evs[t]:
            if e[0] == "OFFLINE":
                i = int(e[1])
                seen[i] = t
            else:
                m = e[1]
                if m == "HERE":
                    for i in range(numberOfUsers):
                        if seen[i] + 60 <= t: mentions[i] += 1
                elif m == "ALL":
                    for i in range(numberOfUsers):
                        mentions[i] += 1
                else:
                    for u in e[1].split(" "):
                        i = int(u[2:])
                        mentions[i] += 1
    return mentions

def countMentions(numberOfUsers: int, events: list[list[str]]) -> list[int]:
    events = sorted(events, key=lambda x: (int(x[1]), x[0] == "MESSAGE"))

    seen = [float("-inf")] * numberOfUsers
    mentions = [0] * numberOfUsers
    for e in events:
        t = int(e[1])
        if e[0] == "OFFLINE":
            i = int(e[2])
            seen[i] = t
        else:
            m = e[2]
            if m == "HERE":
                for i in range(numberOfUsers):
                    if seen[i] + 60 <= t: mentions[i] += 1
            elif m == "ALL":
                for i in range(numberOfUsers):
                    mentions[i] += 1
            else:
                for u in m.split(" "):
                    i = int(u[2:])
                    mentions[i] += 1
    return mentions

#https://leetcode.com/problems/coupon-code-validator/?envType=daily-question&envId=2025-12-13
def validateCoupons(code: list[str], businessLine: list[str], isActive: list[bool]) -> list[str]:
    codes = []
    for i in range(len(code)):
        c1 = code[i] and not search(r"[^A-Za-z0-9_]", code[i])
        c2 = businessLine[i] in {"electronics", "grocery", "pharmacy", "restaurant"}
        c3 = isActive[i]
        if all([c1, c2, c3]): codes.append((businessLine[i], code[i]))
    codes = sorted(codes, key=lambda x: (x[0], x[1]))
    return [x[1] for x in codes]

#https://leetcode.com/problems/number-of-ways-to-divide-a-long-corridor/?envType=daily-question&envId=2025-12-14
def numberOfWays(corridor: str) -> int:
    SCNT = corridor.count("S")
    if not SCNT or SCNT % 2 != 0: return 0

    seg = []
    scnt = pcnt = 0
    for c in corridor:
        if c == "S": 
            scnt += 1
            if pcnt: 
                seg.append(pcnt)
                pcnt = 0
            if scnt == 2:
                seg.append("SEG")
                scnt = 0
        elif not scnt: pcnt += 1
    if scnt: seg.append("SEG")
    elif pcnt: seg.append(pcnt)
    
    k = 1
    for i in range(1, len(seg) - 1):
        if seg[i] != "SEG" and seg[i - 1] == seg[i + 1] == "SEG":
            k *= (seg[i] + 1)
    return k % (10**9 + 7)

#https://leetcode.com/problems/number-of-smooth-descent-periods-of-a-stock/?envType=daily-question&envId=2025-12-15
#spent half an hour to not figure out this easy af formula
def getDescentPeriods(prices: list[int]) -> int:
    p = 1
    k = 1
    for i in range(1, len(prices)):
        if prices[i - 1] - prices[i] == 1: k += 1
        else: k = 1
        p += k
    return p

#https://leetcode.com/problems/maximum-profit-from-trading-stocks-with-discounts/?envType=daily-question&envId=2025-12-16
#this problem is straight up impossible to solve
#pajeets just copied solution from ChatGPT lmao
def maxProfit(n: int, present: list[int], future: list[int], hierarchy: list[list[int]], budget: int) -> int:
    def merge(a, b):
        c = [0] * (budget + 1)
        for i in range(budget + 1):
            best = 0
            for j in range(i + 1):
                v = a[i - j] + b[j]
                if v > best:
                    best = v
            c[i] = best
        return c
    
    g = [[] for _ in range(n)]
    for u, v in hierarchy:
        g[u - 1].append(v - 1)
        
    def dfs(u):
        cost = present[u]
        dcost = cost // 2

        sub0 = [0] * (budget + 1)
        sub1 = [0] * (budget + 1)
        for v in g[u]:
            c0, c1 = dfs(v)
            sub0 = merge(sub0, c0)
            sub1 = merge(sub1, c1)

        dp0 = sub0[:]  # not using discount on u
        dp1 = sub0[:]  # discount still available at u
        gain = future[u]

        for i in range(budget + 1):
            if i >= dcost: dp1[i] = max(dp1[i], sub1[i - dcost] + gain - dcost)
            if i >= cost: dp0[i] = max(dp0[i], sub1[i - cost] + gain - cost)
        return dp0, dp1
    return dfs(0)[0][budget]

#https://leetcode.com/problems/best-time-to-buy-and-sell-stock-v/submissions/1858136503/?envType=daily-question&envId=2025-12-17
#fuck dp, i hate this topic with a burning passion
def maximumProfit(prices: list[int], k: int) -> int:
    d = [[0, -prices[0], prices[0]] for _ in range(k + 1)]
    n = len(prices)

    for i in range(1, n):
        p = prices[i]
        for t in range(k, 0, -1):
            prev = d[t - 1][0]
            d[t][0] = max(d[t][0], d[t][1] + p, d[t][2] - p)
            d[t][1] = max(d[t][1], prev - p)
            d[t][2] = max(d[t][2], prev + p)
    return d[k][0]

#https://leetcode.com/problems/best-time-to-buy-and-sell-stock-using-strategy/?envType=daily-question&envId=2025-12-18
def maxProfit(prices: list[int], strategy: list[int], k: int) -> int:
    n = len(prices)
    p = [prices[0]] * n
    s = [prices[0] * strategy[0]] * n
    for i in range(1, n):
        s[i] = s[i - 1] + strategy[i] * prices[i]
        p[i] = p[i - 1] + prices[i]

    def _chk(i):
        j = i - k #el before current boundary: a[1, 2 | 3, 4], a[j] = 2
        if j < 0: ls = lp = 0
        else: ls, lp = s[j], p[j]

        rs, rp = s[i], p[i] 
        ssum = rs - ls
        modsum = 0 * (p[j + k // 2] - lp) + (rp - p[j + k // 2])
        return modsum - ssum
    
    mxdelta = 0
    for i in range(k - 1, n):
        mxdelta = max(mxdelta, _chk(i))

    if mxdelta == 0: return s[-1]
    else: return s[-1] + mxdelta

def maxProfit(prices: list[int], strategy: list[int], k: int) -> int:
    n = len(prices)
    p = [0] * (n + 1)
    s = [0] * (n + 1)
    for i in range(n):
        s[i + 1] = s[i] + strategy[i] * prices[i]
        p[i + 1] = p[i] + prices[i]

    mxd = 0
    for i in range(n - k + 1):
        old_seg = s[i + k] - s[i]
        new_seg = p[i + k] - p[i + k // 2]
        mxd = max(mxd, new_seg - old_seg)
    return s[-1] + mxd

#https://leetcode.com/problems/find-all-people-with-secret/?envType=daily-question&envId=2025-12-19
#LOST 2 FUCKING HOURS BECAUSE I WAS ASSIGNING PARENT TO CHILD INSTEAD CHILD's PARENT FUUUUUUUUUUU
def findAllPeople(n: int, meetings: list[list[int]], firstPerson: int) -> list[int]:
    meetings = sorted(meetings, key=lambda x: x[2])
    mt = defaultdict(list)
    for m in meetings:
        mt[m[2]].append((m[0], m[1]))

    p = [i for i in range(n + 1)]
    p[firstPerson] = 0
    def _find(x):
        if x == p[x]: return x
        else: 
            p[x] = _find(p[x])
            return p[x]

    for t in mt:
        for a, b in mt[t]:
            pa, pb = _find(a), _find(b)
            if pa != pb:
                if pa == 0: p[pb] = pa #remember this, PARENT TO CHILD's PARENT!!!!!
                elif pb == 0: p[pa] = pb
                else: p[pb] = pa
        for a, b in mt[t]:
            if _find(a) != 0 and _find(b) != 0:
                p[a] = a
                p[b] = b
            else: p[a] = p[b] = 0

    res = [i for i in range(n + 1) if p[i] == 0]
    return res

#https://leetcode.com/problems/delete-columns-to-make-sorted/submissions/1860796138/?envType=daily-question&envId=2025-12-20
def minDeletionSize(strs: list[str]) -> int:
    res = 0
    for x in range(len(strs[0])):
        for y in range(1, len(strs)):
            if strs[y][x] < strs[y - 1][x]: 
                res += 1
                break
    return res

def minDeletionSize(strs: list[str]) -> int:
    mins = list(strs[0])
    res = 0
    for s in strs[1:]:
        for i in range(len(s)):
            if not mins[i]: continue
            if s[i] < mins[i]: 
                mins[i] = None
                res += 1
            else: mins[i] = s[i]
    return res

#https://leetcode.com/problems/delete-columns-to-make-sorted-ii/?envType=daily-question&envId=2025-12-21
def minDeletionSize(strs: list[str]) -> int:
    n, m = len(strs), len(strs[0])
    s = [1] + [False] * (n - 1)
    d = 0

    for x in range(m):
        flag = False
        for y in range(1, n):
            if not s[y] and strs[y][x] < strs[y - 1][x]:
                flag = True
                break
        if flag:
            d += 1
            continue
        
        for y in range(1, n):
            if not s[y] and strs[y][x] > strs[y - 1][x]:
                s[y] = True
        if all(s): break
    return d

#https://leetcode.com/problems/delete-columns-to-make-sorted-iii/?envType=daily-question&envId=2025-12-22
def minDeletionSize(strs: list[str]) -> int:
    m = len(strs[0])
    dp = [1] * m
    for i in range(m - 2, -1, -1):
        for j in range(i + 1, m):
            if all([row[i] <= row[j] for row in strs]): dp[i] = max(dp[i], 1 + dp[j])
    return m - max(dp)

#https://leetcode.com/problems/two-best-non-overlapping-events/?envType=daily-question&envId=2025-12-23
def maxTwoEvents(events: list[list[int]]) -> int:
    evs_s = sorted(events, key=lambda x: x[0])
    evs_e = collections.deque(sorted(events, key=lambda x: x[1]))

    res = max(v for _, _, v in events)
    mxev = 0
    for st, et, ev in evs_s:
        while evs_e and evs_e[0][1] < st:
            #the main trick here is that if evs_e's end time is less than some evs_s's start time, 
            #we can compare it's value to mxev and delete it, because this event will always be available for pair
            #with the next evs_s
            _, _, v = evs_e.popleft()
            mxev = max(mxev, v)
        res = max(res, ev + mxev)
    return res

#https://leetcode.com/problems/apple-redistribution-into-boxes/?envType=daily-question&envId=2025-12-24
#beats 4%😇😇😇
def minimumBoxes(apple: list[int], capacity: list[int]) -> int:
    capacity = sorted(capacity, reverse=True)
    asum = sum(apple)

    psum = capacity.copy()
    for i in range(1, len(capacity)):
        psum[i] += psum[i - 1]

    l, r = 0, len(capacity) - 1
    while l <= r:
        m = (l + r) // 2
        if psum[m] == asum: return m + 1 
        elif psum[m] < asum: l = m + 1
        else: r = m - 1
    return l + 1

def minimumBoxes(apple: list[int], capacity: list[int]) -> int:
    capacity = sorted(capacity, reverse=True)
    asum = sum(apple)
    curr = 0
    for i in range(len(capacity)):
        curr += capacity[i]
        if asum <= curr: return i + 1

#https://leetcode.com/problems/maximize-happiness-of-selected-children/?envType=daily-question&envId=2025-12-25
def maximumHappinessSum(happiness: list[int], k: int) -> int:
    h = sorted(happiness, reverse=True)
    decay = 0   
    res = 0
    for i in range(k):
        v = h[i] - decay
        if v > 0: res += v
        else: break
        decay += 1
    return res

#https://neetcode.io/problems/string-encode-and-decode/history
#https://leetcode.com/problems/encode-and-decode-strings/description/
class Solution:
    def encode(self, strs: list[str]) -> str:
        r = ""
        for s in strs:
            r += f"{str(len(s))}#{s}"
        return r
    
    def decode(self, s: str) -> list[str]:
        r = []
        i = 0
        while i < len(s):
            j = i
            while s[j] != "#": j += 1
            subl = int(s[i:j])
            sub = s[j + 1:j + 1 + subl]
            r.append(sub)
            i = j + subl + 1
        return r

#https://leetcode.com/problems/car-fleet/
def carFleet(target: int, position: list[int], speed: list[int]) -> int:
    z = sorted(zip(position, speed), reverse=True)
    stack = []
    for s, v in z:
        t = (target - s) / v
        if not stack or t > stack[-1]:
            stack.append(t)
    return len(stack)

#https://leetcode.com/problems/largest-rectangle-in-histogram/
#SOLVED MYSELF + BEATS 97.07% GOYDAAAAAAAAA
def largestRectangleArea(heights: list[int]) -> int:
    mx = 0
    stack = collections.deque()
    for h in heights:
        w = 1
        carry_w = 0
        while stack and stack[-1][0] > h:
            ph, pw = stack.pop()
            w += pw
            pw += carry_w
            mx = max(mx, ph * pw)
            carry_w = pw
            
        if stack and stack[-1][0] == h: 
            stack[-1][1] += w
        else:
            stack.append([h, w])

    total_w = 0
    while stack:
        h, w = stack.pop()
        total_w += w
        mx = max(mx, h * total_w)
    return mx

#https://leetcode.com/problems/permutation-in-string/
#solved by myself
def checkInclusion(s1: str, s2: str) -> bool:
    if len(s1) == 1: return s1 in s2
    cnt = collections.Counter(s1)
    chars = set(s1)
    l = 0
    while l < len(s2):
        if s2[l] not in chars: l += 1
        else:
            curr_cnt = collections.Counter()
            curr_cnt[s2[l]] += 1
            r = l + 1
            while r < len(s2):
                c = s2[r]
                if c in chars:
                    curr_cnt[c] += 1
                    while l < r and curr_cnt[c] > cnt[c]:
                        curr_cnt[s2[l]] -= 1
                        l += 1
                else: break

                if curr_cnt == cnt: return True
                else: r += 1
            l = r + 1
    return False

def checkInclusion(s1: str, s2: str) -> bool:
    if len(s1) == 1: return s1 in s2
    if len(s1) > len(s2): return False

    s1cnt, s2cnt = [0] * 26, [0] * 26
    for i in range(len(s1)):
        s1cnt[ord(s1[i]) - ord('a')] += 1
        s2cnt[ord(s2[i]) - ord('a')] += 1
    if s1cnt == s2cnt: return True

    l, r = 0, len(s1)
    while r < len(s2):
        s2cnt[ord(s2[l]) - ord('a')] -= 1
        s2cnt[ord(s2[r]) - ord('a')] += 1
        l += 1
        r += 1
        if s1cnt == s2cnt: return True
    return False

#https://leetcode.com/problems/longest-repeating-character-replacement/
#sovled by myself
def characterReplacement(s: str, k: int) -> int:
    m = 0
    cnt = [0] * 26
    for l in range(len(s)):
        if l > 0: cnt[ord(s[l - 1]) - ord("A")] -= 1
        for r in range(l + m, len(s)):
            nchars = r - l + 1
            cnt[ord(s[r]) - ord("A")] += 1
            if nchars - max(cnt) > k: break
            m = max(m, nchars)
    return m

def characterReplacement(s: str, k: int) -> int:
    cnt = {}
    l, mxf, res = 0, 0, 0
    for r in range(len(s)):
        cnt[s[r]] = 1 + cnt.get(s[r], 0)
        mxf = max(mxf, cnt[s[r]])
        while (r - l + 1) - mxf > k:
            cnt[s[l]] -= 1
            l += 1
        res = max(res, r - l + 1)
    return res   

#https://leetcode.com/problems/sliding-window-maximum/
def maxSlidingWindow(nums: list[int], k: int) -> list[int]:
    h = []
    for i in range(k):
        heappush(h, (-nums[i], i))
    res = [-h[0][0]]
    
    for i in range(k, len(nums)):
        heappush(h, (-nums[i], i))
        while h[0][1] <= i - k: heappop(h)
        res.append(-h[0][0])
    return res

def maxSlidingWindow(nums: list[int], k: int) -> list[int]:
    q = collections.deque()
    for i in range(k):
        while q and nums[i] >= nums[q[-1]]:
            q.pop()
        q.append(i)
    
    res = [nums[q[0]]]
    for i in range(k, len(nums)):
        while q and q[0] <= i - k: q.popleft() #remove largest elements that are out of bounds
        while q and nums[i] >= nums[q[-1]]: q.pop() #remove elements that are smaller than current
        q.append(i)
        res.append(nums[q[0]])
    return res

#https://leetcode.com/problems/binary-search/
#⭐ Somewhat stable version of a binary search
def search(nums: list[int], target: int) -> int:
    l, r = 0, len(nums) - 1
    while l <= r:
        m = l + ((r - l) // 2)
        if nums[m] < target: l = m + 1
        elif nums[m] > target: r = m - 1
        else: return m
    return -1

#https://leetcode.com/problems/time-based-key-value-store/
class TimeMap:

    def __init__(self):
        self.map = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.map[key].append((timestamp, value))

    #O(n) solution is somehow faster than O(log n) bruh
    def get(self, key: str, timestamp: int) -> str:
        values = self.map[key]
        if not values: return ""
        else:
            i = len(values) - 1
            while i >= 0:
                if values[i][0] <= timestamp: return values[i][1]
                else: i -= 1
            return ""

    def get(self, key: str, timestamp: int) -> str:
        values = self.map[key]
        if not values: return ""
        else:
            l, r = 0, len(values) - 1 #finding position where number should be placed
            while l <= r:
                m = l + ((r - l) // 2)
                if values[m][0] == timestamp: return values[m][1]
                elif values[m][0] > timestamp: r = m - 1
                elif values[m][0] < timestamp: l = m + 1
            return values[l - 1][1] if l > 0 else "" 
        
#https://leetcode.com/problems/reorder-list/
#beats 5.16% Xd
def reorderList(head: ListNode) -> None:
    nodes = []
    while head:
        node = head
        head = head.next
        node.next = None
        nodes.append(node)
    
    res = head = ListNode()
    left = True
    while nodes:
        node = None
        if left: node = nodes.pop(0)
        else: node = nodes.pop()

        head.next = node
        head = node
        left = not left

    return res.next

def reorderList(head: ListNode) -> None:
    slow = fast = head
    while fast and fast.next: #find middle
        slow = slow.next
        fast = fast.next.next
    
    end = slow.next
    prev = slow.next = None 
    while end: #reverse second half
        temp = end.next
        end.next = prev
        prev = end
        end = temp
    
    start, end = head, prev
    while end: #rearrange list
        stemp, etemp = start.next, end.next
        end.next = None
        start.next = end
        start.next.next = stemp

        end = etemp
        start = start.next.next
    return head

#https://leetcode.com/problems/find-the-duplicate-number/
#we can't modify initial array, so this solution is somewhat wrong
def findDuplicate(nums: list[int]) -> int:
    if len(nums) == 1: return nums[0]
    nums = sorted(nums)
    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1]: return nums[i]

def findDuplicate(nums: list[int]) -> int:
    slow = fast = 0
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast: break

    slow2 = 0
    while True:
        slow = nums[slow]
        slow2 = nums[slow2]
        if slow == slow2: return slow

#https://leetcode.com/problems/lru-cache/
class LRUCache:
    class Node:
        def __init__(self, key, val):
            self.key = key
            self.val = val
            self.prev = self.next = None

    def __init__(self, capacity: int):
        self.cap = capacity
        self.map = {}
 
        self.tail, self.head = self.Node(0, 0), self.Node(0, 0) #dummy nodes
        self.tail.next = self.head
        self.head.prev = self.tail

    def _remove(self, node):
        prev, next = node.prev, node.next
        prev.next, next.prev = next, prev

    def _insert(self, node):
        prev, next = self.head.prev, self.head #remember that head is a dummy node
        prev.next = next.prev = node
        node.prev, node.next = prev, next

    def get(self, key: int) -> int:
        if key in self.map:
            node = self.map[key]
            self._remove(node)
            self._insert(node)
            return node.val
        else: return -1

    def put(self, key: int, value: int) -> None:
        if key in self.map:
            self._remove(self.map[key])

        self.map[key] = self.Node(key, value)
        self._insert(self.map[key]) 
        if len(self.map) > self.cap:
            true_tail = self.tail.next
            self._remove(true_tail)
            del self.map[true_tail.key]

#https://leetcode.com/problems/lru-cache/
class LRUCache:
    def __init__(self, capacity: int):
        self.map = collections.OrderedDict()
        self.cap = capacity

    def get(self, key: int) -> int:
        if key not in self.map: return -1
        self.map.move_to_end(key)
        return self.map[key]

    def put(self, key: int, value: int) -> None:
        if key in self.map:
            self.map.move_to_end(key)
        self.map[key] = value

        if len(self.map) > self.cap:
            self.map.popitem(last=False)

#https://leetcode.com/problems/combinations/
def combine(n: int, k: int) -> list[list[int]]:
    res = []
    curr = []
    def _comb(i):
        if i == n + 1:
            if len(curr) == k: res.append(curr.copy())
            return

        curr.append(i)
        _comb(i + 1)
        curr.pop()
        _comb(i + 1)
    _comb(1)
    return res

def combine(n: int, k: int) -> list[list[int]]:
    res = []
    curr = []
    def _comb(start):
        if len(curr) == k:
            res.append(curr.copy())
            return 

        for i in range(start, n + 1):
            curr.append(i)
            _comb(i + 1)
            curr.pop()
    _comb(1)
    return res

#https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
def lowestCommonAncestor(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    res = None
    def _dfs(node):
        nonlocal res
        if not node: return

        fl = node == p or node == q
        l, r = _dfs(node.left), _dfs(node.right)
        if (fl and (l or r)) or (l and r): 
            res = node
            return False
        return fl or l or r
    _dfs(root)
    return res

def lowestCommonAncestor(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    def _dfs(node):
        if max(p.val, q.val) < node.val: return _dfs(node.left)
        elif min(p.val, q.val) > node.val: return _dfs(node.right)
        else: return node
    return _dfs(root)

def lowestCommonAncestor(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    while True:
        if max(p.val, q.val) < root.val: root = root.left
        elif min(p.val, q.val) > root.val: root = root.right
        else: return root

#https://leetcode.com/problems/subtree-of-another-tree/
def isSubtree(root: TreeNode, subRoot: TreeNode) -> bool:
    q = [root]
    chk_roots = []
    while q:
        node = q.pop(0)
        if node.val == subRoot.val: chk_roots.append(node)

        if node.left: q.append(node.left)
        if node.right: q.append(node.right)
    
    def _dfs(node, cornode):
        if not node and not cornode: return True
        if (not node) ^ (not cornode): return False
        if node.val != cornode.val: return False

        return _dfs(node.left, cornode.left) and _dfs(node.right, cornode.right)
    
    return any(_dfs(node, subRoot) for node in chk_roots)

#https://leetcode.com/problems/balanced-binary-tree/
def isBalanced(root: TreeNode) -> bool:
    res = True
    def _dfs(node):
        nonlocal res
        if not node: return 0
        if not res: return 0

        l, r = _dfs(node.left), _dfs(node.right)
        if abs(l - r) > 1: res = False
        return 1 + max(l, r)
    _dfs(root)
    return res

#https://leetcode.com/problems/same-tree/
def isSameTree(p: TreeNode, q: TreeNode) -> bool:
    def _dfs(node, cornode): 
        if node == cornode == None: return True
        if (not node) ^ (not cornode): return False
        if node.val != cornode.val: return False
        if node.val == cornode.val: 
            return _dfs(node.left, cornode.left) and _dfs(node.right, cornode.right)
    return _dfs(p, q)

#https://leetcode.com/problems/binary-tree-maximum-path-sum/
#SOLVED IT MYSELF FIRST TRY (IDEA)
#just use Kadane's algorithm
def maxPathSum(root: TreeNode) -> int:
    mx = float("-inf")
    def _dfs(node):
        nonlocal mx
        if not node: return 0

        l, r = _dfs(node.left), _dfs(node.right)
        pmx = max(l, r)
        curr = max(node.val, node.val + pmx)
        mx = max(mx, curr, node.val + l + r)
        return curr
    _dfs(root)
    return mx

#https://leetcode.com/problems/serialize-and-deserialize-binary-tree/
class Codec:
    def serialize(self, root):
        s = "" 
        if not root: return s
        q = [root]
        while q:
            node = q.pop(0)
            if node: 
                s += f"{node.val}|" #use "".join() instead concatenation (I'm too lazy Xd)
                q.append(node.left)
                q.append(node.right)
            else: s += "#|"
        return s

    def deserialize(self, data): 
        vals = data.split("|")[:-1]
        if not vals: return None
        root = TreeNode(int(vals.pop(0)))
        q = [root]
        while vals:
            node = q.pop(0)
            if vals: 
                v = vals.pop(0)
                if v != "#": 
                    node.left = TreeNode(int(v))
                    q.append(node.left)
            if vals:
                v = vals.pop(0)
                if v != "#": 
                    node.right = TreeNode(int(v))
                    q.append(node.right)
        return root
    
#https://leetcode.com/problems/last-stone-weight/
def lastStoneWeight(stones: list[int]) -> int:
    h = []
    for s in stones: heappush(h, -s)

    while len(h) > 1:
        y, x = abs(heappop(h)), abs(heappop(h))
        if x == y: continue
        else: heappush(h, -(y - x))
    return -max(h) if h else 0

#https://leetcode.com/problems/kth-largest-element-in-a-stream/
class KthLargest:
    def __init__(self, k: int, nums: list[int]):
        self.k = k
        self.mxh = []
        self.mnh = []
        
        nums = sorted(nums, reverse=True)
        for n in nums[:k-1]: heappush(self.mnh, n)
        for n in nums[k-1:]: heappush(self.mxh, -n)

    def add(self, val: int) -> int:
        if self.mnh and val >= self.mnh[0]: 
            kth = heappushpop(self.mnh, val)
            heappush(self.mxh, -kth)
            return kth
        else:
            heappush(self.mxh, -val)
            return -self.mxh[0] 
    
    #second variant
    def __init__(self, k: int, nums: list[int]):
        self.h, self.k = nums, k
        heapify(self.h)
        while len(self.h) > self.k: heappop(self.h)

    def add(self, val: int) -> int:
       heappush(self.h, val)
       while len(self.h) > self.k: heappop(self.h)
       return self.h[0]

#https://leetcode.com/problems/k-closest-points-to-origin/
def kClosest(points: list[list[int]], k: int) -> list[list[int]]:
    h = []
    for p in points:
        heappush(h, (-dist(p, [0, 0]), p))
        while len(h) > k: heappop(h)
    return [p for _, p in h]

#https://leetcode.com/problems/task-scheduler/
def leastInterval(tasks: list[str], n: int) -> int:
    h = []
    for t in set(tasks): heappush(h, -tasks.count(t))

    q = collections.deque()
    t = 0
    while h or q:
        t += 1
        if not h: t = q[0][1]
        else:
            rem = heappop(h) + 1 #-(rem - 1)
            if rem: q.append((rem, t + n))
        if q and q[0][1] == t: heappush(h, q.popleft()[0]) #when cooldown fress up
    return t

def leastInterval(tasks: list[str], n: int) -> int:
    cnt = collections.Counter(tasks)
    
    mx = max(cnt.values())
    mx_cnt = list(cnt.values()).count(mx)

    #(mx - 1) gaps with (n + 1) slots each + mx_cnt unused elements at the end
    t = (mx - 1) * (n + 1) + mx_cnt
    return max(t, len(tasks))

#https://leetcode.com/problems/design-twitter/
class Twitter:
    def __init__(self):
        self.u_tweets = defaultdict(list)
        self.u_subs = defaultdict(set)
        self.clock = 0

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.u_tweets[userId].append((-self.clock, tweetId))
        self.clock += 1

    def getNewsFeed(self, userId: int) -> list[int]:
        posts = self.u_tweets[userId].copy()
        for uid in self.u_subs[userId]:
            posts.extend(self.u_tweets[uid])
        heapify(posts)

        feed = []
        while posts and len(feed) < 10: feed.append(heappop(posts)[1])
        return feed

    def follow(self, followerId: int, followeeId: int) -> None:
        self.u_subs[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        self.u_subs[followerId] -= {followeeId}

class Twitter:
    def __init__(self):
        self.u_tweets = defaultdict(list)
        self.u_subs = defaultdict(set)
        self.clock = 0

    def postTweet(self, userId: int, tweetId: int) -> None:
        tweets = self.u_tweets[userId]
        tweets.append((-self.clock, tweetId))
        if len(tweets) > 10: tweets.pop(0)
        self.clock += 1

    def getNewsFeed(self, userId: int) -> list[int]:
        res = []
        resh = [] #heap by the most recent one (max heap)

        subs = self.u_subs[userId]
        subs.add(userId)
        h = [] #heap by the least recent one (min heap)
        for uid in subs:
            tweets = self.u_tweets[uid]
            if not tweets: continue
            i = len(tweets) - 1
            time, tid = tweets[i]

            heappush(h, (-time, tid, uid, i))
            if len(h) > 10: heappop(h) #pop the least recent tweet (that's why we append -time)

        while h: #construct heap by the most recent one (max heap)
            time, tid, uid, i = heappop(h)
            heappush(resh, (-time, tid, uid, i)) #the most recent is at resh[0] (max heap)
        while resh and len(res) < 10:
            time, tid, uid, i = heappop(resh)
            res.append(tid)
            if i > 0: #if there are more tweets by tid user (i > 0), add the previous tweet
                time, tid = self.u_tweets[uid][i - 1] 
                heappush(resh, (time, tid, uid, i - 1))
        return res

    def follow(self, followerId: int, followeeId: int) -> None:
        self.u_subs[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        self.u_subs[followerId] -= {followeeId}

#https://leetcode.com/problems/find-median-from-data-stream/
class MedianFinder:
    def __init__(self):
        self.l = [] #max heap
        self.r = [] #min heap

    def addNum(self, num: int) -> None:
        llen, rlen = len(self.l), len(self.r)
        if not self.r: heappush(self.r, num)
        elif llen <= rlen: #makes that so median is in the left side if llen + rlen is odd
            heappush(self.r, num)
            heappush(self.l, -heappop(self.r))
        else:
            heappush(self.l, -num)
            heappush(self.r, -heappop(self.l))  

    def findMedian(self) -> float:
        llen, rlen = len(self.l), len(self.r)
        if llen == rlen: return (-self.l[0] + self.r[0]) / 2
        elif self.l: return -self.l[0]
        else: return self.r[0]

#https://leetcode.com/problems/jump-game-ii/
def jump(nums: list[int]) -> int:
    n = len(nums)
    k = 0
    l = r = 0
    while r < n - 1:
        mxd = 0 #amongst nodes that you can reach from l (left)
        for i in range(l, r + 1): mxd = max(mxd, i + nums[i])
        l = r + 1
        r = mxd
        k += 1
    return k

#https://leetcode.com/problems/gas-station/
#beats 5%😭😭😭😭
def canCompleteCircuit(gas: list[int], cost: list[int]) -> int:
    n = len(gas)
    balance = []
    for i in range(n): 
        b = gas[i] - cost[i]
        if b >= 0: heappush(balance, (-b, i))
    if balance and balance[0][0] == 0 and max(gas) < max(cost): return -1

    while balance:
        _, idx = heappop(balance)
        t = gas[idx]
        for i in range(n):
            t -= cost[(idx + i) % n]
            if t < 0: break
            t += gas[(idx + i + 1) % n]
        else: return idx
    return -1 

def canCompleteCircuit(gas: list[int], cost: list[int]) -> int:
    if sum(gas) < sum(cost): return -1
    
    t = 0
    idx = 0
    for i in range(len(gas)):
        t += gas[i] - cost[i]
        if t < 0: 
            t = 0
            idx = i + 1
    return idx

#https://leetcode.com/problems/hand-of-straights/description/
#beats 5% again Xd
def isNStraightHand(hand: list[int], groupSize: int) -> bool:
    if len(hand) % groupSize != 0: return False

    h = []
    for c in set(hand): heappush(h, (c, hand.count(c)))
    
    while h:
        rem = []
        prev = None
        for _ in range(groupSize):
            if not h: return False
            c, cnt = heappop(h)
            if (prev == None) or (c - prev == 1): prev = c
            else: return False
            if cnt > 1: rem.append((c, cnt - 1))
        for el in rem: heappush(h, el)
    return True

def isNStraightHand(hand: list[int], groupSize: int) -> bool:
    if len(hand) % groupSize != 0: return False

    c = collections.Counter(hand)
    for n in hand:
        st = n
        while c[st - 1]: st -= 1 #smallest element in the hand
        while st <= n: 
            while c[st]: #create sequence starting from this element
                for i in range(st, st + groupSize):
                    if not c[i]: return False
                    c[i] -= 1
            st += 1
    return True

#https://leetcode.com/problems/merge-triplets-to-form-target-triplet/
def mergeTriplets(triplets: list[list[int]], target: list[int]) -> bool:
    at, bt, ct = target
    h = []
    for a, b, c in triplets:
        if (a > at) or (b > bt) or (c > ct): continue
        else: h.append([a, b, c])

    if not h: return False
    ar, br, cr = h.pop(0)
    for a, b, c in h:
        ar, br, cr = max(ar, a), max(br, b), max(cr, c)
    return (ar == at) and (br == bt) and (cr == ct) 

def mergeTriplets(triplets: list[list[int]], target: list[int]) -> bool:
    at, bt, ct = target
    a = b = c = False
    for a1, b1, c1 in triplets:
        a |= (a1 == at and b1 <= bt and c1 <= ct)
        b |= (a1 <= at and b1 == bt and c1 <= ct)
        c |= (a1 <= at and b1 <= bt and c1 == ct)
        if a and b and c: return True
    return False

#https://leetcode.com/problems/partition-labels/
def partitionLabels(s: str) -> list[int]:
    cnt = collections.Counter(s)
    res = []

    curr_cnt = defaultdict(int)
    for c in s:
        curr_cnt[c] += 1
        if all(cnt[c] == curr_cnt[c] for c in curr_cnt):
            res.append(sum(curr_cnt.values()))
            curr_cnt = defaultdict(int)
    return res

def partitionLabels(s: str) -> list[int]:
    last = {}
    for c in set(s): last[c] = s.rfind(c)

    res = []
    size = end = 0
    for i in range(len(s)):
        size += 1
        end = max(end, last[s[i]])
        if i == end:
            res.append(size)
            size = end = 0
    return res

#https://leetcode.com/problems/valid-parenthesis-string/
#at least I was close
def checkValidString(s: str) -> bool:
    psum = stars = 0
    for c in s:
        if c == "(": psum += 1
        elif c == "*": stars += 1
        else: psum -= 1

        if psum < 0: 
            if abs(psum) > stars: return False
            else: 
                stars -= abs(psum)
                psum = 0
    
    psum = stars = 0
    for c in s[::-1]:
        if c == "(": psum -= 1
        elif c == "*": stars += 1
        else: psum += 1

        if psum < 0: 
            if abs(psum) > stars: return False
            else: 
                stars -= abs(psum)
                psum = 0
    return True

def checkValidString(s: str) -> bool:
    lmax = lmin = 0 #max and min unmatched "("
    for c in s:
        if c == "(": 
            lmax += 1
            lmin += 1
        elif c == ")":
            lmax -= 1
            lmin -= 1
        else:
            lmax += 1 # ((* -> (((
            lmin -= 1 # ((* -> (()
        
        if lmax < 0: return False
        lmin = max(lmin, 0)
    return lmin == 0

#https://leetcode.com/problems/insert-interval/
def insert(intervals:list[list[int]], newInterval: list[int]) -> list[list[int]]:
    idx = bisect.bisect_left(intervals, newInterval)
    intervals.insert(idx, newInterval)
    
    for i in range(idx - 1, -1, -1):
        if intervals[i][1] >= newInterval[0]:
            newInterval[0] = intervals[i][0]
            newInterval[1] = max(newInterval[1], intervals[i][1])
            intervals[i] = None
    
    for i in range(idx + 1, len(intervals)):
        if intervals[i][0] <= newInterval[1]:
            newInterval[1] = max(newInterval[1], intervals[i][1])
            intervals[i] = None

    return [x for x in intervals if x]

#https://leetcode.com/problems/meeting-rooms/
def canAttendMeetings(intervals: list[Interval]) -> bool:
    if not intervals: return True
    t = sorted(intervals, key=lambda x: (x.start, x.end))
    curre = t.pop(0).end
    for intv in t:
        if intv.start < curre: return False
        else: curre = intv.end
    return True

#https://neetcode.io/problems/meeting-schedule-ii/question
def minMeetingRooms(intervals: list[Interval]) -> int:
    if not intervals: return 0
    t = sorted([(x.start, x.end) for x in intervals])
    h = [t.pop(0)[1]]
    for s, e in t:
        chk = []
        while h:
            pe = heappop(h)
            if pe > s: chk.append(pe)
            else: 
                chk.append(e)
                break
        else: chk.append(e)
        for e in chk: heappush(h, e)
    return len(h)

def minMeetingRooms(intervals: list[Interval]) -> int:
    t = sorted([(x.start, x.end) for x in intervals])
    h = []
    for s, e in t:
        if h and h[0] <= s: heappop(h)
        heappush(h, e)
    return len(h)

def minMeetingRooms(intervals: list[Interval]) -> int:
    d = defaultdict(int)
    for t in intervals:
        d[t.start] += 1
        d[t.end] -= 1
    
    curr = res = 0
    for k in sorted(d.keys()):
        curr += d[k]
        res = max(res, curr)
    return res

#https://leetcode.com/problems/minimum-interval-to-include-each-query/
def minInterval(intervals: list[list[int]], queries: list[int]) -> list[int]:
    t = sorted(intervals)
    h = []
    res = {}
    i = 0
    for q in sorted(queries):
        while i < len(t) and t[i][0] <= q: #by start
            heappush(h, [t[i][1] - t[i][0] + 1, t[i][1]])
            i += 1
        
        while h and h[0][1] < q: heappop(h) #by end
        res[q] = h[0][0] if h else -1
    return [res[q] for q in queries]

#https://leetcode.com/problems/palindrome-partitioning/
def partition(s: str) -> list[list[str]]:
    def _chk(l, r):
        while l < r:
            if s[l] == s[r]:
                l += 1
                r -= 1
            else: return False
        return True
    
    res = []
    curr = []
    def _partition(i):
        if i >= len(s): 
            res.append(curr.copy())
            return 

        for r in range(len(s) - 1, i - 1, -1):
            if _chk(i, r): 
                curr.append(s[i:r+1])
                _partition(r + 1)
                curr.pop()
    _partition(0)
    return res

#https://leetcode.com/problems/n-queens/
def solveNQueens(n: int) -> list[list[str]]:
    def _fill(x, y, f):
        f["x"].add(x)
        f["y"].add(y)
        def _elph_fill(x, y, dx, dy):
            if not (0 <= x < n) or not (0 <= y < n): return

            f["coords"].add((x, y))
            _elph_fill(x + dx, y + dy, dx, dy)

        _elph_fill(x, y, -1, 1)
        _elph_fill(x, y, 1, 1)
        _elph_fill(x, y, -1, -1)
        _elph_fill(x, y, 1, -1)
    
    res = []
    curr = {}
    def _dfs(rem, f):
        if rem == 0:
            r = []
            for y, x in sorted(curr.items()):
                r.append("."*x + "Q" + "."*(n - x - 1))
            res.append(r)
            return 
        
        y = n - rem
        for x in range(n):
            if (x not in f["x"]) and (y not in f["y"]) and ((x, y) not in f["coords"]):
                fc = copy.deepcopy(f)
                _fill(x, y, fc)

                curr[y] = x
                _dfs(rem - 1, fc)
                curr[y] = None
                    
    _dfs(n, defaultdict(set))
    return res

def solveNQueens(n: int) -> list[list[str]]:
    res = []
    curr = {}
    d1, d2, xf = set(), set(), set()
    def _dfs(rem):
        if rem == 0:
            r = []
            for y, x in sorted(curr.items()):
                r.append("."*x + "Q" + "."*(n - x - 1))
            res.append(r)
            return 
        
        y = n - rem
        for x in range(n):
            if (x not in xf) and ((y + x) not in d1) and ((y - x) not in d2):
                xf.add(x) 
                d1.add(y + x) #apparently, you can store diagonals like this
                d2.add(y - x) #UPD: row - col (y - x) is always a constant

                curr[y] = x
                _dfs(rem - 1)
                curr[y] = None

                xf.remove(x)
                d1.remove(y + x)
                d2.remove(y - x)       
    _dfs(n)
    return res

#https://leetcode.com/problems/walls-and-gates/description/
def islandsAndTreasure(grid: list[list[int]]) -> None:
    w, h = len(grid[0]), len(grid)
    for y in range(h):
        for x in range(w):
            if grid[y][x] == 0:
                q = collections.deque([(x, y, 0)])
                chk = set()
                while q:
                    x, y, d = q.popleft()
                    for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                        nx, ny = x + dx, y + dy
                        if not (0 <= nx < w) or\
                            not (0 <= ny < h) or\
                            grid[ny][nx] == 0 or\
                            grid[ny][nx] == -1 or\
                            (nx, ny) in chk: continue
                        grid[ny][nx] = min(grid[ny][nx], d + 1)
                        q.append((nx, ny, d + 1))
                        chk.add((nx, ny))

#bfs, but starts search from land node, which is more efficient
def islandsAndTreasure(grid: list[list[int]]) -> None:
    w, h = len(grid[0]), len(grid)
    INF = 2**31 - 1
    def _bfs(x, y):
        chk = set([(x, y)])
        q = collections.deque([(x, y)])
        d = 0
        while q:
            for _ in range(len(q)):
                x, y = q.popleft()
                if grid[y][x] == 0: return d
                for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < w) or not (0 <= ny < h) or (nx, ny) in chk or grid[ny][nx] == -1: continue
                    chk.add((nx, ny))
                    q.append((nx, ny))
            d += 1
        return INF

    for y in range(h):
        for x in range(w):
            if grid[y][x] == INF: 
                grid[y][x] = _bfs(x, y)

#https://leetcode.com/problems/pacific-atlantic-water-flow
#beats 5% Xd
def pacificAtlantic(heights: list[list[int]]) -> list[list[int]]:
    w, h = len(heights[0]), len(heights)
    def _bfs(x, y):
        p = a = False
        chk = set([(x, y)])
        q = collections.deque([(x, y)])
        while q:
            x, y = q.popleft()
            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                nx, ny = x + dx, y + dy
                if (ny < 0 or nx < 0): 
                    p = True
                    continue
                if (ny >= h or nx >= w): 
                    a = True
                    continue
                if ((nx, ny) in chk) or (heights[ny][nx] > heights[y][x]): continue
                chk.add((nx, ny))
                q.append((nx, ny))
        return p and a
       
    res = []
    for y in range(h):
        for x in range(w):
            if _bfs(x, y): res.append([y, x])
    return res

def pacificAtlantic(heights: list[list[int]]) -> list[list[int]]:
    w, h = len(heights[0]), len(heights)
    pt = [[False] * w for _ in range(h)]
    at = [[False] * w for _ in range(h)]

    def _bfs(q, chkt):
        while q:
            x, y = q.popleft()
            chkt[y][x] = True
            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w) or\
                not (0 <= ny < h) or\
                chkt[ny][nx] or\
                (heights[ny][nx] < heights[y][x]): 
                    continue
                q.append((nx, ny))
    
    pq, aq = collections.deque(), collections.deque()
    for x in range(w): #bordering tiles
        pq.append((x, 0)) 
        aq.append((x, h - 1))
    for y in range(h):
        pq.append((0, y))
        aq.append((w - 1, y))
    
    _bfs(pq, pt)
    _bfs(aq, at)
    
    res = []
    for y in range(h):
        for x in range(w):
            if pt[y][x] and at[y][x]: res.append([y, x])
    return res

#https://leetcode.com/problems/graph-valid-tree/description/
def validTree(n: int, edges: list[list[int]]) -> bool:
    inc, outc = defaultdict(list), defaultdict(list)
    for p, c in edges:
        inc[c].append(p)
        outc[p].append(c)

    chkn = chkrel = None
    for node in range(len(edges)):
        if chkn == None:
            if len(inc[node]) <= 1 and len(outc[node]) > 1: 
                chkn = node
                chkrel = inc
            elif len(inc[node]) > 1 and len(outc[node]) <= 1: #upside-down
                chkn = node
                chkrel = outc
        if len(inc[node]) > 1 and len(outc[node]) > 1: return False
    
    if chkn == None: #random bs, go😭 
        chkn = 0
        chkrel = inc

    v = set()
    while chkrel[chkn]: #chkn = head of the tree
        if chkn in v: return False
        if len(chkrel[chkn]) > 1: return False

        v.add(chkn)
        chkn = chkrel[chkn][0]
    
    chkrel = inc if chkrel != inc else outc #change direction
    v = set()
    def _dfs(node):
        if node in v: return False
        v.add(node)
        if not chkrel[node]: return True

        return all(_dfs(snode) for snode in chkrel[node])

    return _dfs(chkn) and len(v) == n #in case of separate trees

def validTree(n: int, edges: list[list[int]]) -> bool:
    if len(edges) > n - 1: return False

    rel = defaultdict(list)
    for a, b in edges:
        rel[a].append(b)
        rel[b].append(a)

    v = set()
    def _dfs(node, parent):
        if node in v: return False
        
        v.add(node)
        for snode in rel[node]:
            if snode == parent: continue
            if not _dfs(snode, node): return False
        return True

    return _dfs(0, -1) and len(v) == n 

def validTree(n: int, edges: list[list[int]]) -> bool:
    if len(edges) > n - 1: return False

    p = list(range(n)) 
    p_pwr = [1] * n
    def _find(x):
        if p[x] == x: return x
        else: 
            p[x] = _find(p[x])
            return p[x]
    
    for a, b in edges:
        pa, pb = _find(a), _find(b)
        if pa == pb: return False

        if p_pwr[pb] > p_pwr[pa]: pa, pb = pb, pa
        p[pb] = pa
        p_pwr[pa] += p_pwr[pb]
        p_pwr[pb] = 0 
        n -= 1
    return n == 1

#https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/description/
def countComponents(n: int, edges: list[list[int]]) -> int:
    p = list(range(n))
    def _find(x):
        if p[x] == x: return x
        else:
            p[x] = _find(p[x])
            return p[x]

    for a, b in edges:
        pa, pb = _find(a), _find(b)
        if pa != pb:
            p[pb] = pa
            n -= 1
    return n

def countComponents(n: int, edges: list[list[int]]) -> int:
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    v = set()
    def _dfs(node):
        if node in v: return
        if not adj[node]: return

        v.add(node)
        for snode in adj[node]: _dfs(snode)
    
    k = 0
    for node in range(n):
        if node in v: continue
        else:
            _dfs(node)
            k += 1
    return k

#https://leetcode.com/problems/word-ladder/
#beats 5%💀
def ladderLength(beginWord: str, endWord: str, wordList: list[str]) -> int:
    adj = defaultdict(list)
    wpool = set([beginWord, *wordList])
    for w1, w2 in itertools.combinations(wpool, 2):
        d = 0
        for c1, c2 in zip(w1, w2):
            if c1 != c2: d += 1
            if d >= 2: break
        else:
            adj[w1].append(w2)
            adj[w2].append(w1)
    
    chk = set(beginWord)
    q = collections.deque([(beginWord, 1)])
    while q: 
        w1, k = q.popleft()
        if w1 == endWord: return k
        for w2 in adj[w1]:
            if w2 not in chk:
                q.append((w2, k + 1))
                chk.add(w2)
    return 0

def ladderLength(beginWord: str, endWord: str, wordList: list[str]) -> int:
    adj = defaultdict(list)
    for w in wordList:
        for i in range(len(w)): adj[w[:i] + "_" + w[i + 1:]].append(w)
    
    chk = set([beginWord])
    q = collections.deque([(beginWord, 1)])
    while q:
        w, k = q.popleft()
        if w == endWord: return k
        
        for i in range(len(w)):
            for nw in adj[w[:i] + "_" + w[i + 1:]]:
                if nw not in chk: 
                    chk.add(w)
                    q.append((nw, k + 1))
    return 0

#https://leetcode.com/problems/network-delay-time/
def networkDelayTime(times: list[list[int]], n: int, k: int) -> int:
    rel = defaultdict(list)
    for u, v, t in times: heappush(rel[u], (t, v))
    
    time = defaultdict(lambda: float("inf"))
    q = [(0, k)]
    while q:
        node_t, node = heappop(q)
        while rel[node]:
            snode_t, snode = heappop(rel[node])
            if snode == k: continue

            t = node_t + snode_t
            time[snode] = min(time[snode], t)
            heappush(q, (t, snode))
    return max(time.values()) if len(time) == n - 1 else -1

#dijkstra
def networkDelayTime(times: list[list[int]], n: int, k: int) -> int:
    rel = defaultdict(list)
    for u, v, t in times: rel[u].append((t, v))
    
    t = 0
    v = set()
    q = [(0, k)]
    while q:
        node_t, node = heappop(q)
        if node in v: continue
        v.add(node)
        t = node_t
        for snode_t, snode in rel[node]:
            if snode in v: continue
            heappush(q, (t + snode_t, snode))
            
    return t if len(v) == n else -1

#https://leetcode.com/problems/min-cost-to-connect-all-points/
#prim's with heap o algo
def minCostConnectPoints(points: list[list[int]]) -> int:
    total_d = 0
    dist = {}

    h = [(0, 0, 0)]
    while h:
        d, n, pn = heappop(h)
        if n in dist: continue

        dist[n] = d
        total_d += dist[n]
        for sn in range(len(points)):
            if sn in dist: continue
            sd = abs(points[n][0] - points[sn][0]) + abs(points[n][1] - points[sn][1])
            heappush(h, (sd, sn, n))
    return total_d

#prim's algorithm
def minCostConnectPoints(points: list[list[int]]) -> int:
    total_d = 0
    dist = defaultdict(lambda: float("inf"))
    v = set()
    node = 0
    for _ in range(len(points) - 1):
        v.add(node)
        nxt = None
        for snode in range(len(points)):
            if snode in v: continue
            d = abs(points[node][0] - points[snode][0]) + abs(points[node][1] - points[snode][1]) 
            dist[snode] = min(dist[snode], d)
            if nxt == None or dist[snode] < dist[nxt]: nxt = snode

        total_d += dist[nxt]
        node = nxt
    return total_d

def minCostConnectPoints(points: list[list[int]]) -> int:
    p = list(range(len(points)))
    p_pwr = [1] * len(points)
    def _find(x):
        if p[x] == x: return x
        else:
            p[x] = _find(p[x])
            return p[x]

    h = []
    for a in range(len(points)):
        x1, y1 = points[a]
        for b in range(a + 1, len(points)):
            x2, y2 = points[b]
            heappush(h, (abs(x2 - x1) + abs(y2 - y1), a, b))
                    
    res = 0
    while h:
        d, a, b = heappop(h)
        pa, pb = _find(a), _find(b)
        if pa != pb:
            if p_pwr[pb] > p_pwr[pa]: pa, pb = pb, pa
            p[pb] = pa
            p_pwr[pa] += p_pwr[pb]
            p_pwr[pb] = 0

            res += d
    return res

#https://leetcode.com/problems/cheapest-flights-within-k-stops/
def findCheapestPrice(n: int, flights: list[list[int]], src: int, dst: int, k: int) -> int:
    rel = defaultdict(list)
    for a, b, w in flights: rel[a].append((b, w))

    h = [(0, -1, src)]
    dist = defaultdict(lambda: float("inf"))
    while h:
        tw, ck, a = heappop(h)
        if dist[a] < ck: continue #if there exists a path with fewer steps
        if a == dst: return tw

        dist[a] = ck
        for b, bw in rel[a]:
            if dist[b] < dist[a] + 1: continue #if there exists a path with fewer steps, again
            if ck + 1 > k: continue
            heappush(h, (tw + bw, ck + 1, b))
    return -1

#https://leetcode.com/problems/swim-in-rising-water/
def swimInWater(grid: list[list[int]]) -> int:
    W, H = len(grid[0]), len(grid)
    def _get(x, y):
        r = []
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W) or not (0 <= ny < H): continue
            r.append((grid[ny][nx], nx, ny)) 
        return r

    dist = {}
    h = [(grid[0][0], 0, 0)]
    while h:
        tw, x, y = heappop(h)
        if (x == W - 1) and (y == H - 1): return tw
        if (x, y) in dist: continue

        dist[(x, y)] = tw
        for aw, ax, ay in _get(x, y):
            if (ax, ay) in dist: continue
            heappush(h, (max(tw, aw), ax, ay))

#https://leetcode.com/problems/reconstruct-itinerary/
#i'm clueless
def findItinerary(tickets: list[list[str]]) -> list[str]:
    tickets = sorted(tickets)
    rel = defaultdict(list)
    for a, b in tickets: rel[a].append(b)

    s = ["JFK"] #memory o algo
    res = []
    while s:
        curr = s[-1]
        if not rel[curr]: res.append(s.pop()) #if all path are used o algo
        else: s.append(rel[curr].pop(0))
    return res[::-1]

#TLE on leetcode
def findItinerary(tickets: list[list[str]]) -> list[str]:
    tickets = sorted(tickets)
    adj = defaultdict(list)
    for a, b in tickets: adj[a].append(b)

    res = ["JFK"]
    def _dfs(n):
        if len(res) == len(tickets) + 1: return True
        if not adj[n]: return False

        for _ in range(len(adj[n])):
            sn = adj[n].pop(0)
            res.append(sn)
            if _dfs(sn): return True
            res.pop()
            adj[n].append(sn)
        return False
    _dfs("JFK")
    return res

#https://leetcode.com/problems/alien-dictionary/
def foreignDictionary(words: list[str]) -> str:
    adj = {c: set() for w in words for c in w}
    for i in range(1, len(words)):
        w1, w2 = words[i - 1], words[i]
        mnl = min(len(w1), len(w2))
        if len(w1) > len(w2) and w1[:mnl] == w2[:mnl]: return ""
        for j in range(mnl): 
            if w1[j] != w2[j]: 
                adj[w1[j]].add(w2[j])
                break
    
    res = []
    v = {}
    def dfs(n): #cycle detection
        if n in v: return v[n]

        v[n] = True
        for sn in adj[n]:
            if dfs(sn): return True
        v[n] = False
        res.append(n)
    
    for n in adj: 
        if dfs(n): return ""
    return "".join(res[::-1])

def foreignDictionary(words: list[str]) -> str:
    adj = {c: set() for w in words for c in w}
    indegree = {c: 0 for c in adj} #inc
    for i in range(1, len(words)):
        w1, w2 = words[i - 1], words[i]
        mnl = min(len(w1), len(w2))
        if len(w1) > len(w2) and w1[:mnl] == w2[:mnl]: return ""
        for j in range(mnl): 
            if w1[j] != w2[j]: 
                if w2[j] not in adj[w1[j]]:
                    adj[w1[j]].add(w2[j])
                    indegree[w2[j]] += 1
                break

    res = []
    q = collections.deque([c for c in indegree if indegree[c] == 0])
    while q:
        n = q.popleft()
        res.append(n)
        for sn in adj[n]:
            indegree[sn] -= 1
            if indegree[sn] == 0: q.append(sn)
    if len(res) != len(indegree): return ""
    return "".join(res)

#https://leetcode.com/problems/decode-ways/
def numDecodings(s: str) -> int:
    n, n1, n2 = 0, 1, 1 #fibonacci o algo
    for i in range(len(s) - 1, -1, -1):
        if s[i] == "0": n = 0
        else: n = n1
        if i + 1 < len(s) and ((s[i] == "1") or (s[i] == "2" and s[i + 1] in "0123456")): 
            n += n2
        n, n1, n2 = 0, n, n1
    return n1

#https://leetcode.com/problems/maximum-product-subarray/
#brutforce
def maxProduct(nums: list[int]) -> int:
    res = float("-inf")
    for i in range(len(nums)):
        if nums[i] in [0, 1]: 
            res = max(res, nums[i])
            continue
        p = 1
        
        for j in range(i, len(nums)):
            p *= nums[j]
            res = max(res, p)
    return res

def maxProduct(nums: list[int]) -> int:
    res = float("-inf")
    
    cmn = cmx = 1 #mx and min prod at the current index
    for n in nums:
        t = cmn * n
        cmn = min(t, cmx * n, n)
        cmx = max(t, cmx * n, n)
        res = max(cmn, cmx, res)
    return res

#https://leetcode.com/problems/partition-equal-subset-sum/
def canPartition(nums: list[int]) -> bool:
    sm = sum(nums)
    if sm % 2 != 0: return False #we need 2 subsets smh

    @cache
    def _dfs(i, t):
        if i >= len(nums): return t == 0
        if t < 0: return False

        return _dfs(i + 1, t) or _dfs(i + 1, t - nums[i])

    return _dfs(0, sm // 2)

def canPartition(nums: list[int]) -> bool:
    sm = sum(nums)
    if sm % 2 != 0: return False #we need 2 subsets smh

    t = sm // 2
    d = set([0])

    for i in range(len(nums)):
        nd = set()
        for pt in d: 
            if pt + nums[i] == t: return True
            nd |= {pt, pt + nums[i]}
        d = nd
    return False

#https://leetcode.com/problems/house-robber-ii/
#fuck DP
def rob(nums: list[int]) -> int:
    def _r(nums):
        r2 = r1 = 0
        for n in nums:
            nxt = max(n + r2, r1)
            r2, r1 = r1, nxt
        return r1
    return max(nums[0], _r(nums[1:]), _r(nums[:-1]))

#https://leetcode.com/problems/counting-bits/submissions/1972705203/
def countBits(n: int) -> list[int]:
    r = []
    
    for i in range(n + 1):
        k = 0
        while i:
            k += i % 2
            i //= 2
        r.append(k)
    return r

def countBits(n: int) -> list[int]:
    d = [0] * (n + 1)
    p = 1 #minimum value for current slots: 100 = 4, 1000 = 8, 10000 = 16, etc.
    for i in range(1, n + 1):
        if p * 2 == i: p = i
        d[i] = 1 + d[i - p]
    return d

#https://leetcode.com/problems/reverse-bits/
def reverseBits(n):
    res = 0
    for _ in range(32):
        res = (res << 1) | (n & 1)
        n >>= 1
    return res

#https://leetcode.com/problems/sum-of-two-integers/
def getSum(a: int, b: int) -> int:
    MAXINT = 0x7FFFFFFF
    m = 0xFFFFFFFF
    while b:
        rem = (a & b) << 1
        a = (a ^ b) & m #"& m": cutting bits that are out of 32-bit range
        b = rem & m
    return a if a <= MAXINT else ~(a ^ m) #python doesn't understand that leftomst bit is a sign of negative number, so ~(a ^ m)
    #the same code in C++:
    #int getSum(int a, int b) {
    #   while(b != 0) {
    #       int carry = (a & b) << 1; 
    #       a = a ^ b;               
    #       b = carry;               
    #   }
    #   return a;
    #}

#https://leetcode.com/problems/reverse-integer/
def reverse(self, x: int) -> int:
    sign = -1 if x < 0 else 1
    x = abs(x)
    n = 0
    while x:
        n *= 10
        n += x % 10
        x //= 10
    if -2**31 <= sign * n <= 2**31 - 1: return sign * n
    else: return 0

#https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/
#i have no idea how i was able to solve this first try o algo
def maxProfit(prices: list[int]) -> int:
    @cache
    def d(i, prev):
        if i >= len(prices): return 0

        m = 0
        if prev == "buy": m = max(prices[i] + d(i + 2, "sell"), d(i + 1, "buy"))
        else: m = max(-prices[i] + d(i + 1, "buy"), d(i + 1, None))
        return m
    return d(0, None)

def maxProfit(prices: list[int]) -> int:
    d = [[0] * 2 for _ in range(len(prices) + 2)]
    for i in range(len(prices) - 1, -1, -1):
        d[i][0] = max(-prices[i] + d[i + 1][1], d[i + 1][0]) #sell
        d[i][1] = max(prices[i] + d[i + 2][0], d[i + 1][1]) #buy
    return d[0][0]

#https://leetcode.com/problems/target-sum/
def findTargetSumWays(nums: list[int], target: int) -> int:
    d = [defaultdict(int) for _ in range(len(nums) + 1)]
    d[0][0] = 1
    for i in range(len(nums)):
        for sm, cnt in d[i].items():
            d[i + 1][sm - nums[i]] += cnt
            d[i + 1][sm + nums[i]] += cnt
    return d[len(nums)][target]

#https://leetcode.com/problems/longest-increasing-path-in-a-matrix/
#I was close at least
def longestIncreasingPath(matrix: list[list[int]]) -> int:
    w, h = len(matrix[0]), len(matrix)
    r = [[0] * w for _ in range(h)] 
    q = collections.deque()
    for y in range(h):
        for x in range(w):
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w) or not (0 <= ny < h): continue
                if matrix[ny][nx] > matrix[y][x]: r[y][x] += 1 #cells that have higher value
            if r[y][x] == 0: q.append((x, y))

    mx = 0
    while q:
        for _ in range(len(q)):
            x, y = q.popleft()
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w) or not (0 <= ny < h): continue
                if matrix[ny][nx] < matrix[y][x]:
                    r[ny][nx] -= 1 #if (x, y) had higher value
                    if r[ny][nx] == 0: q.append((nx, ny))
        mx += 1
    return mx

def longestIncreasingPath(matrix: list[list[int]]) -> int:
    from functools import cache
    w, h = len(matrix[0]), len(matrix)
    @cache
    def _dfs(x, y, pv):
        if not (0 <= x < w) or not (0 <= y < h) or matrix[y][x] <= pv: return 0
        
        res = 1
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            res = max(res, 1 + _dfs(nx, ny, matrix[y][x]))
        return res
    
    mx = 0
    for y in range(h):
        for x in range(w):
            mx = max(mx, _dfs(x, y, -1))
    return mx

#https://leetcode.com/problems/distinct-subsequences/
#😭😭😭
def numDistinct(s: str, t: str) -> int:
    if len(t) > len(s): return 0

    @cache
    def _dfs(i, j):
        if j == len(t): return 1
        if i == len(s): return 0

        res = _dfs(i + 1, j)
        if s[i] == t[j]: res += _dfs(i + 1, j + 1)
        return res
    return _dfs(0, 0)

#https://leetcode.com/problems/interleaving-string/
def isInterleave(s1: str, s2: str, s3: str) -> bool:
    if len(s1) + len(s2) > len(s3): return False
    @cache
    def _dfs(i, j, k):
        if k == len(s3): return True
        
        r = False
        if (i < len(s1)) and (s1[i] == s3[k]): r |= _dfs(i + 1, j, k + 1)
        if (j < len(s2)) and (s2[j] == s3[k]): r |= _dfs(i, j + 1, k + 1)
        return r
    return _dfs(0, 0, 0)

#https://leetcode.com/problems/regular-expression-matching/
def isMatch(s: str, p: str) -> bool:
    @cache
    def _dfs(i, j):
        if j == len(p): return i == len(s)

        match = i < len(s) and (s[i] == p[j] or p[j] == ".")
        if (j + 1) < len(p) and p[j + 1] == "*": return _dfs(i, j + 2) or (match and _dfs(i + 1, j)) # don't use or use
        if match: return _dfs(i + 1, j + 1)
        return False

    return _dfs(0, 0)

#https://leetcode.com/problems/burst-balloons/
def maxCoins(nums: list[int]) -> int:
    nums = [1] + nums + [1]
    @cache
    def _dfs(l, r):
        if l > r: return 0

        mx = 0
        for i in range(l, r + 1):
            coins = nums[l - 1] * nums[i] * nums[r + 1] #reconstruct balloons
            coins += _dfs(l, i - 1) + _dfs(i + 1, r)
            mx = max(mx, coins)
        return mx   
    return _dfs(1, len(nums) - 2)

#https://leetcode.com/problems/coin-change-ii/
def change(amount: int, coins: list[int]) -> int:
    @cache
    def _dfs(i, rem):
        if i >= len(coins): return rem == 0
        if rem < 0: return 0
        if rem == 0: return 1

        return _dfs(i, rem - coins[i]) + _dfs(i + 1, rem)
    return _dfs(0, amount)

def change(amount: int, coins: list[int]) -> int:
    d = [1] + [0] * amount
    for i in range(len(coins)):
        nd = d.copy()
        for a in range(1, amount + 1):
            if a - coins[i] >= 0: nd[a] += nd[a - coins[i]]
        d = nd
    return d[amount]

def change(amount: int, coins: list[int]) -> int:
    d = [1] + [0] * amount
    for c in coins:
        for a in range(c, amount + 1): 
            d[a] += d[a - c]
    return d[amount]

#https://leetcode.com/problems/happy-number/submissions/1987955743/
def isHappy(n: int) -> bool:
    def _sq(n):
        r = 0
        while n:
            r += (n % 10) ** 2
            n //= 10
        return r

    slow = fast = n
    while True:
        slow = _sq(slow)
        fast = _sq(_sq(fast))
        if slow == fast: return fast == 1

def isHappy(n: int) -> bool:
    def _sq(n):
        r = 0
        while n:
            r += (n % 10) ** 2
            n //= 10
        return r

    v = set()
    while n not in v:
        v.add(n)
        n = _sq(n)
        if n == 1: return True
    return False

#https://leetcode.com/problems/detect-squares/
class DetectSquares:
    def __init__(self):
        self.x = defaultdict(list)
        self.y = defaultdict(list)
        self.xy = defaultdict(int)

    def add(self, point: list[int]) -> None:
        self.x[point[0]].append(point[1])
        self.y[point[1]].append(point[0])
        self.xy[tuple(point)] += 1

    def count(self, point: list[int]) -> int:
        px, py = point
        dist = defaultdict(int)
        for y in self.x[px]:
            if y == py: continue 
            dist[abs(py - y)] += 1 #bottom
        for x in self.y[py]: 
            if x == px: continue
            dist[abs(px - x)] += 1 #left

        k = 0
        for d in dist:
            if dist[d] <= 1: continue
            for kx, ky in [(1, 1), (1, -1), (-1, -1), (-1, 1)]:
                nx, ny = kx * d + px, ky * d + py
                k += self.xy[(nx, py)] * self.xy[(px, ny)] * self.xy[(nx, ny)]
        return k

class DetectSquares:
    def __init__(self):
        self.points = []
        self.p_cnt = defaultdict(int)

    def add(self, point: list[int]) -> None:
        self.points.append(point)
        self.p_cnt[tuple(point)] += 1

    def count(self, point: list[int]) -> int:
        k = 0
        px, py = point
        for x, y in self.points: #search for diagonal point
            if (abs(px - x) != abs(py - y)) or (px == x) or (py == y): continue
            k += self.p_cnt[(x, py)] * self.p_cnt[(px, y)] #no need for self.p_cnt[(x, y)], because duplicate diag points are included in the cycle
        return k

#https://leetcode.com/problems/multiply-strings/
def multiply(num1: str, num2: str) -> str:
    conv = {str(x): x for x in range(10)}
    num1, num2 = sorted([num1, num2], key=len, reverse=True)
    res = 0
    for i in range(len(num1) - 1, -1, -1):
        a = conv[num1[i]]
        row = 0
        carry = 0
        for j in range(len(num2) - 1, -1, -1):
            m = a * conv[num2[j]] + carry
            row = (m % 10) * 10**(len(num2) - j - 1) + row
            carry = m // 10
        row = carry * 10**len(num2) + row
        row *= 10**(len(num1) - i - 1)
        res += row
    return str(res)

def multiply(num1: str, num2: str) -> str:
    if (num1 == "0") or (num2 == "0"): return "0"
    res = [0] * (len(num1) + len(num2))
    num1, num2 = num1[::-1], num2[::-1]
    for i in range(len(num1)):
        for j in range(len(num2)):
            m = int(num1[i]) * int(num2[j])
            res[i + j] += m
            res[i + j + 1] += res[i + j] // 10
            res[i + j] %= 10

    res = res[::-1]
    i = 0
    while i < len(res) and res[i] == 0: i += 1
    res = map(str, res[i:])
    return "".join(res)