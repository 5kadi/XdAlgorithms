from math import inf 

def create_array_variation(arr: list) -> tuple:
    return (arr, sorted(arr))

ARR1, ARR1_SORTED = create_array_variation([1, 3, 9, 17, 6, 4, 35, 9, 11, 0, 53, 7, 1])

ARR2, ARR2_SORTED = create_array_variation([3, 1, 2])

GRAPH1 = [
    [0, 3, 1, 3, inf, inf],
    [3, 0, 4, inf, inf, inf],
    [1, 4, 0, inf, 7, 5],
    [3, inf, inf, 0, inf, 2],
    [inf, inf, 7, inf, 0, 4],
    [inf, inf, 5, 2, 4, 0]
]

GRAPH2 = [
    [0, 2, inf, 3, 1, inf, inf, 10],
    [2, 0, 4, inf, inf, inf, inf, inf],
    [inf, 4, 0, inf, inf, inf, inf, 3],
    [3, inf, inf, 0, inf, inf, inf, 8],
    [1, inf, inf, inf, 0, 2, inf, inf],
    [inf, inf, inf, inf, 2, 0, 3, inf],
    [inf, inf, inf, inf, inf, 3, 0, 1],
    [10, inf, 3, 8, inf, inf, 1, 0],
]

GRAPH3 = [
    [[0,0,1], [20,0,1], [30,0,1], [10,0,1], [0,0,1]], #[0, 1, 2]: 0 - weight, 1 - denominator, 2 - direction (+/-)
    [[20,0,-1], [0,0,1], [40,0,1], [0,0,1], [30,0,1]],
    [[30,0,-1], [40,0,-1], [0,0,1], [10,0,1], [20,0,1]],
    [[10,0,-1], [0,0,1], [10,0,-1], [0,0,1], [20,0,1]],
    [[0,0,1], [30,0,-1], [20,0,-1], [20,0,-1], [0,0,1]],
]


KNAPSACK_ITEMS1 = (
    (95, 55),
    (4, 10),
    (60, 47),
    (32, 5),
    (23, 4),
    (72, 50),
    (80, 8),
    (62, 61),
    (65, 85),
    (46, 87)
)
KNAPSACK_ITEMS2 = (
    (10, 60),
    (20, 100),
    (30, 120)
)