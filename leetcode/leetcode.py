from heapq import *
import collections
from math import inf, ceil, floor, log2



class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Interval:
    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end

class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

class GraphNode:
    def __init__(self, val: int = 0, neighbors: list['GraphNode'] = []):
        self.val = val
        self.neighbors = neighbors 



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

#https://leetcode.com/problems/single-number/description/
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
def reverseList(head: ListNode):
    nxt = None
    while head:
        temp = head.next
        head.next = nxt
        nxt = head
        head = temp
    return nxt

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
    res = 0
    def _get_longest(root):
        if not root:
            return 0

        l = _get_longest(root.left)
        r = _get_longest(root.right)

        nonlocal res
        res = max(res, l + r)

        return 1 + max(l, r)

    _get_longest(root)
    return res

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

#https://leetcode.com/problems/spiral-matrix/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
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
            heapq.heappop(min_heap)

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

#https://leetcode.com/problems/number-of-islands/solutions/6744132/video-check-4-directions-bonus-solutions/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def numIslands(grid: list[list[str]]) -> int:
    from collections import deque
    islands = 0
    visited = set()
    rows, cols = len(grid), len(grid[0])

    def bfs(r, c):
        q = deque()
        visited.add((r, c))
        q.append((r, c))

        while q:
            row, col = q.popleft()
            directions = [[1,0],[-1,0],[0,1],[0,-1]]

            for dr, dc in directions:
                r, c = row + dr, col + dc
                if 0 <= r < rows and 0 <= c < cols and grid[r][c] == "1" and (r, c) not in visited:
                    q.append((r, c))
                    visited.add((r, c))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1" and (r, c) not in visited:
                islands += 1
                bfs(r, c)

    return islands

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

#https://leetcode.com/problems/subsets/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
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

#https://leetcode.com/problems/top-k-frequent-elements/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100
def topKFrequent(nums: list[int], k: int) -> list[int]:
    count = {}
    for n in nums:
        count[n] = 1 + count.get(n, 0)
    
    freq = [[] for _ in range(len(nums) + 1)]
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
    row = [1] * n #n = w, this is 'second' row 
    for i in range(1, m):
        for j in range(1, n):
            right = row[j - 1]
            up = row[j]
            row[j] = right + up
    return row[-1]

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
    inf = amount + 1
    memo = [0] + [inf]*amount

    for i in range(1, amount + 1):
        for c in coins:
            if i >= c:
                memo[i] = min(memo[i], memo[i - c] + 1)
    
    if memo[-1] == inf:
        return -1
    else:
        return memo[-1]
    
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

#https://leetcode.com/problems/course-schedule/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-1
def canFinish(numCourses: int, prerequisites: list[list[int]]) -> bool:
    from collections import defaultdict
    req = defaultdict(list)
    for c, r in prerequisites:
        req[c].append(r)

    used = set()
    def _dfs(c):
        if not req[c]:
            return True
        if c in used:
            return False
        
        used.add(c)
        for r in req[c]:
            if not _dfs(r):
                return False
    
        req[c] = []
        return True

    for i in range(numCourses):
        if not _dfs(i):
            return False
    return True

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
    """
    if len(nums) <= 1:
        return True

    nlen = len(nums)
    memo = [False]*len(nums)
    dist = 1
    for i in range(len(nums) - 2, -1, -1):
        if nums[i] < dist:
            dist += 1
            continue
        else:
            memo[i] = True
            dist = 1
    return memo[0]
    """

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
    memo = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                memo[i] = max(memo[j] + 1, memo[i])
    return max(memo)

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
    #explanation: https://www.youtube.com/watch?v=xMBwzNvXmms
    m = len(nums1)
    n = len(nums2)
    if m > n:
        return findMedianSortedArrays(nums2, nums1)
    
    l = 0
    r = m
    while l <= r:
        p1 = (l + r) // 2 #something something make logic consistent
        p2 = (m + n + 1) // 2 - p1 #make logic consistent

        l_p1 = float('-inf') if p1 == 0 else nums1[p1 - 1]
        l_p2 = float('-inf') if p2 == 0 else nums2[p2 - 1]
        r_p1 = float('inf') if p1 == m else nums1[p1]
        r_p2 = float('inf') if p2 == n else nums2[p2]

        if r_p2 >= l_p1 and r_p1 >= l_p2:
            if not (m + n) % 2:
                return (max(l_p1, l_p2) + min(r_p1, r_p2)) / 2
            else:
                return float(max(l_p1, l_p2))
        elif l_p2 > r_p1:
            l = p1 + 1
        else:
            r = p1 - 1

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

#https://leetcode.com/problems/reverse-nodes-in-k-group/solutions/6896538/video-recursive-pattern-bonus-coding-with-iterative-pattern/?utm_source=instabyte.io&utm_medium=referral&utm_campaign=interview-master-100           
def reverseKGroup(head: ListNode, k: int) -> ListNode:
    def _get_kth(node, k):
        while node and k > 0:
            node = node.next
            k -= 1
        return node

    start = dum = ListNode(0, head)
    while True:
        end = _get_kth(start, k)
        if not end:
            break
        group_next = end.next 

        node = start.next 
        nxt = end.next 
        while node != group_next:
            temp = node.next 
            node.next = nxt 
            nxt = node
            node = temp
        temp = start.next
        start.next = end 
        start = temp
    return dum.next

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
def rangeBitwiseAnd(self, left: int, right: int) -> int:
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
def maxSubArray(nums: list[int]) -> int:
    dp = [0] * len(nums)
    dp[0] = nums[0]
    for i in range(1, len(nums)):
        dp[i] = max(nums[i] + dp[i - 1], nums[i])
    return max(dp)

def maxSubArray(nums: list[int]) -> int:
    curr = nums[0]
    mx = nums[0]
    for n in nums[1:]:
        curr += n
        if curr < n:
            curr = n
        #or: curr = max(curr + n, n)
        mx = max(mx, curr)
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

#https://leetcode.com/problems/edit-distance/?envType=study-plan-v2&envId=leetcode-75
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

#https://leetcode.com/problems/search-suggestions-system/?envType=study-plan-v2&envId=leetcode-75
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
    queue = collections.deque()
    m = len(grid)
    n = len(grid[0])
    ones = 0

    for y in range(m):
        for x in range(n):
            if grid[y][x] == 2:
                queue.append((x, y))
            elif grid[y][x] == 1:
                ones += 1
    if not queue and ones == 0:
        return 0
    
    def _fill(x, y):
        nonlocal queue, ones, grid
        for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            nx = x + dx
            ny = y + dy
            if not (0 <= nx < n) or not (0 <= ny < m): continue
            elif grid[ny][nx] == 1:
                ones -= 1
                grid[ny][nx] = 2
                queue.append((nx, ny)) 
    minutes = -1
    while queue:
        minutes += 1
        for _ in range(len(queue)):
            x, y = queue.popleft()
            _fill(x, y)
        
    if ones == 0: return minutes
    else: return -1

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