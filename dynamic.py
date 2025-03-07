from math import inf
memo = {}

#минимальное кол-во монеток, из которого можно составить сумму m
def min_coins(coins: list, m: int) -> int:
    if m in memo:
        return memo[m]
    
    if m == 0:
        ans = 0 #0, потому что кол-во монеток увеличивается в min()
    else:
        ans = inf
        for c in coins:
            sub_m = m - c
            if sub_m < 0:
                continue
            ans = min(ans, min_coins(coins, sub_m) + 1) #вот здесь
    memo[m] = ans
    return ans

#сколькими способами можно составить сумму m из набора монеток
def combo_coins(coins: list, m: int) -> int:
    if m in memo:
        return memo[m]
    
    if m == 0:
        return 1 #1, потому что приведение к сумме 0 считается за комбо
    else:
        combos = 0
        for c in coins:
            sub_m = m - c
            if sub_m < 0:
                continue
            combos += combo_coins(coins, sub_m)
    memo[m] = combos
    return combos

#можно двигаться вправо и влево, изначальная позиция - (0, 0) (левый верхний угол)
#сколькими способами можно достигнуть позиции последней позиции (правый нижний угол)
def maze_problem(w: int, h: int) -> int:
    if (w, h) in memo:
        return memo[(w, h)]
    
    if w == 1 or h == 1:
        return 1 #так же считается за комбо
    else:
        combos = 0
        for move in ["r", "d"]:
            if (move == "r") and w > 1:
                combos += maze_problem(w - 1, h)
            elif (move == "d") and h > 1:
                combos += maze_problem(w, h - 1)
    memo[(w, h)] = combos
    return combos

#какие предметы нужно прихватизировать и на какую максимальную сумму
"""
def knapsack_problem(w: int, table: tuple[tuple]) -> tuple:
    if w in memo:
        return memo[w]
    
    else:
        profit = 0
        combo = []
        for i in range(len(table)):
            sub_w = w - table[i][0] 
            if sub_w < 0:
                continue
            elif sub_w == 0:
                profit = table[i][1]
                combo = [table[i][0]]
                break
            else:
                sub_res = knapsack_problem(sub_w, tuple([x for x in table if x != table[i]]))
                if sub_res[0] + table[i][1] > profit:
                    profit = sub_res[0] + table[i][1]
                    combo = [*sub_res[1], table[i][0]]
    memo[w] = (profit, combo)
    return (profit, combo)
"""

def knapsack_problem_classic(w: int, table: tuple[tuple]) -> int:
    if w in memo:
        return memo[w]
    
    profit = 0
    for weight, value in table:
        sub_w = w - weight
        if sub_w < 0:
            continue
        elif sub_w == 0:
            profit = value
            break
        else:
            profit = max(profit, knapsack_problem_classic(sub_w, set([x for x in table if x != (weight, value)])) + value)
    
    memo[w] = profit
    return profit

#фибоначчи в пару строк
def fibonacci(n: int) -> int:
    memo = {
        1: 1,
        2: 1
    }
    def _fib(n: int) -> int:
        if n in memo:
            return memo[n]
        memo[n] = _fib(n - 1) + _fib(n - 2)
        return memo[n]
    return _fib(n)



#самая длинная последовательность
"""
def longest_subsequence(a: list, b: list) -> tuple:
    a, b = set(a), set(b)
    res = a & b
    return (res, len(res))
"""
def longest_sseq(a: list, b: list) -> int:
    A, B = tuple(a), tuple(b)
    def _longest_sseq(m: int, n: int) -> int:
        if m == 0 or n == 0:
            return 0
        
        res = 0
        if A[m - 1] != B[n - 1]:
            res = max(_longest_sseq(m - 1, n), _longest_sseq(m, n - 1))
        elif A[m - 1] == B[n - 1]:
            res = _longest_sseq(m - 1, n - 1) + 1

        return res
    res = _longest_sseq(len(A), len(B))
    return res                                                                        

#==ХОД КОНЁМ== в какой последовательности конь должен ходить по доске, чтобы побывать в каждой клетке
def knight_problem(n: int) -> list:
    MOVES = (
        (2, 1), (1, 2), 
        (-1, 2), (-2, 1), 
        (-2, -1), (-1, -2), 
        (1, -2), (2, -1) #векторы должны быть расположены так хд (т.н. порядок "одал-свастон")
    )
    board = [[-1]*n for i in range(n)]
    board[0][0] = 0

    def _is_safe(x: int, y: int) -> bool:
        if (x >= 0 and y >= 0 and x < n and y < n and board[y][x] == -1):
            return True
        else:
            return False
        
    def _solve(x: int, y: int, pos: int) -> bool:
        if (pos == n**2):
            return True
        
        for move in MOVES:
            new_x = x + move[0]
            new_y = y + move[1]
            if _is_safe(new_x, new_y):
                board[new_y][new_x] = pos
                if _solve(new_x, new_y, pos + 1):
                    return True
                board[new_y][new_x] = -1
        return False
    
    _solve(0, 0, 1)

    return board