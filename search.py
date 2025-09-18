from arrays import ARR1_SORTED as arr

def binary_search(arr: list, val: int, start: int, end: int) -> int:
    mid = (start + end) // 2

    if arr[mid] == val:
        return mid
    
    if start == end:
        return None
    
    if arr[mid] < val:
        return binary_search(arr, val, mid + 1, end)
    elif arr[mid] > val:
        return binary_search(arr, val, start, mid - 1)
    
def binary_search_bisect(arr: list, val: int, start: int, end: int) -> int:
    from bisect import bisect_left
    idx = bisect_left(arr[start:end+1], val)
    return start + idx
   
def direct_search(arr: list, mask: list) -> int:
    found = None
    shift = 0
    while not found:
        for i in range(len(mask)):
            if arr[shift + i] == mask[i]:
                if i == len(mask) - 1:
                    found = shift
                continue
            else:
                shift += 1
                break 
    return found

#KMP
# 1. p_massive 
# 2. search
# ---
# p[i] - длиннейший перфикс, соответствующий данному суффиксу 
# p[i - 1] - конец длиннейшего префикса, соответствующего данному суффиксу
def p_massive(mask: list) -> list:
    length = len(mask)
    p = [0] * length

    i = 1 
    j = 0

    while i < length:
        if mask[i] == mask[j]:
            p[i] = j + 1
            i += 1
            j += 1
        else:
            if j == 0:
                p[i] = 0
                i += 1
            else:
                j = p[j - 1]
    return p
            
def KMP(arr: list, mask: list) -> int:
    p = p_massive(mask)
    found = None

    j = i = 0

    while i < len(arr):
        if mask[j] == arr[i]:
            if j == len(mask) - 1:
                found = i - j
                break
            i += 1
            j += 1
        else:
            if j > 0:
                j = p[j - 1] #тыкать позиции, пока не будет 0 в худшем случае
            else:
                i += 1
                if i == len(arr) - 1:
                    break

    return found

#BMP
# 1. create an array of a symbol shifts from the last symbol (NOTE: length - 1 - i)
# 2. BMH itself (too hard to describe in english, transitioning to rus):
#   1) проверка конечных символов маски и строки с учётом смещения (i, NOTE: arr[i + j])
#   2F) символ не совпадает? -> 
#       1) добавить к смещению table[mask[j]], если j =/= len(mask) - 1, или table[arr[j]] || table['*']
#       2) ресет указателя маски (j, j = len(mask) - 1)
#   2T) символ совпадает? -> 
#       1) проверять предыдущие символы (j -= 1)
#       2T) found = i; break;
#   3) повторять до тех пор, пока смещение меньше индексов строки (i < len(arr) - 1)

def shift_table(arr: list) -> dict:
    length = len(arr)
    i = length - 2

    shift_table = {}

    while i >= 0:
        if arr[i] not in shift_table.keys():
            shift_table[arr[i]] = length - 1 - i 
        i -= 1
    
    if arr[length - 1] not in shift_table.keys():
        shift_table[arr[length - 1]] = length

    shift_table['*'] = length

    return shift_table

def BMH(arr: list, mask: list) -> int:
    shifts = shift_table(mask)
    found = None
    j = len(mask) - 1
    i = 0

    while i < (len(arr) - 1):
        if arr[i + j] == mask[j]:
            j -= 1
            if j == 0:
                found = i 
                break
        else:
            if j == (len(mask) - 1):
                i += shifts.get(arr[j], False) or shifts['*']
                #print(i)
            else:
                i += shifts[mask[j]]
            j = len(mask) - 1
    return found


    
    
