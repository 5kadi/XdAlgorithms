
#subsets
def subsets(nums: list[int]) -> list[list[int]]:
    res = []
    subset = []

    def create_subset(i):
        if i == len(nums): #and len(subset) == n if we want combs
            res.append(subset[:])
            return
        
        subset.append(nums[i])
        create_subset(i + 1)

        subset.pop()
        create_subset(i + 1)

    create_subset(0)
    return res

#no duplicates
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

def subsets(nums: list[int]) -> list[list[int]]:
    res = [[]]
    for n in nums:
        res += [sub + [n] for sub in res]
    return res

#permutations
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