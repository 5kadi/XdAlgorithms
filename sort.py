from search import binary_search_bisect
from arrays import ARR1

def binary_insertion_sort_bisect(arr: list[int]) -> list[int]:
    for i in range(1, len(arr)):
        val = arr[i]
        j = binary_search_bisect(arr, val, 0, i-1)
        arr = arr[:j] + [val] + arr[j:i] + arr[i+1:]
        #print(arr, i, val, j, sep="\t")
    return arr 


def binary_insertion_sort(arr: list[int]) -> list[int]:
	
    def _binary_search(val: int, start: int, end: int) -> int:
        
        # we need to distinguish whether we 
        # should insert before or after the 
        # left boundary. imagine [0] is the last 
        # step of the binary search and we need 
        # to decide where to insert -1
        if start == end:
            if arr[start] > val:
                return start
            else:
                return start + 1

        # this occurs if we are moving 
        # beyond left's boundary meaning 
        # the left boundary is the least 
        # position to find a number greater than val
        if start > end:
            return start

        mid = (start + end) // 2
        if arr[mid] < val:
            return _binary_search(val, mid + 1, end)
        elif arr[mid] > val:
            return _binary_search(val, start, mid - 1)
        else:
            return mid

    for i in range(1, len(arr)):
        val = arr[i]
        j = _binary_search(val, 0, i - 1)
        arr = arr[:j] + [val] + arr[j:i] + arr[i+1:]

    return arr

def bubble_sort(arr: list[int]) -> list[int]:
    for n in range(len(arr) - 1, 0, -1):
        swap = False
        for i in range(n):
            if arr[i] > arr[i + 1]:
                temp = arr[i]
                arr[i] = arr[i + 1]
                arr[i + 1] = temp
                swap = True
        if not swap:
            break
    return arr

def insertion_sort(arr: list[int]) -> list[int]:
    for i in range(1, len(arr)):
        for j in range(i):  
            if arr[i] < arr[j]:
                #temp = arr[i]
                #arr[i] = arr[j]
                #arr[j] = temp
                arr = arr[:j] + [arr[i]] + arr[j:i] + arr[i+1:]
    return arr


def quick_sort(arr: list[int]) -> list[int]:
    if len(arr) <= 1:
        return arr 
    
    pivot = arr[0]
    l = [i for i in arr[1:] if i < pivot]
    r = [i for i in arr[1:] if i >= pivot]
    return quick_sort(l) + [pivot] + quick_sort(r) 

def merge_sort(arr: list) -> list:
    def _merge(arr1: list, arr2: list) -> list:
        res_arr = []
        i, j = 0, 0
        while i < len(arr1) and j < len(arr2):
            if arr1[i] <= arr2[j]:
                res_arr.append(arr1[i])
                i += 1
            else:
                res_arr.append(arr2[j])
                j += 1
        res_arr += arr1[i:] + arr2[j:]
        return res_arr
    
    if len(arr) == 1:
        return arr
        
    mid = len(arr) // 2
    arr1, arr2 = arr[:mid], arr[mid:]

    if len(arr1) > 1:
        arr1 = merge_sort(arr[:mid])
    if len(arr2) > 1:
        arr2 = merge_sort(arr[mid:])

    res_arr = _merge(arr1, arr2)
    return res_arr








