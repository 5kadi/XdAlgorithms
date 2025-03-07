from search import binary_search_bisect
from arrays import ARR1 as arr 

def insertion_sort(arr: list) -> list:
    for i in range(1, len(arr)):
        val = arr[i]
        j = binary_search_bisect(arr, val, 0, i-1)
        arr = arr[:j] + [val] + arr[j:i] + arr[i+1:]
        #print(arr, i, val, j, sep="\t")
    return arr 

def quick_sort(arr: list) -> list:
    if len(arr) <= 1:
        return arr 
    
    pivot = arr[0]
    l = [i for i in arr[1:] if i < pivot]
    r = [i for i in arr[1:] if i >= pivot]
    return quick_sort(l) + [pivot] + quick_sort(r) 

res = quick_sort(arr)
print(res)





