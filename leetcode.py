class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next





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
def mergeTwoLists(self, list1, list2):
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




