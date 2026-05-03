from math import inf
import collections
import heapq


#Dijkstra
#используется, если требуется найти путь из одной вершины ко всем остальным
#лень объяснять 
def min_index(table: list, checked: set) -> int:
    idx = - 1
    lowest = inf 

    for i, w in enumerate(table):
        if (w < lowest) and (i not in checked):
            idx = i
            lowest = w

    return idx

def dijkstra(graph: list, start: int, end: int) -> int:
    graph_length = len(graph)

    table = [inf]*graph_length
    relations = [0]*graph_length

    i = start
    checked = {i}
    table[i] = 0
       
    while i != -1:
        for j, w in enumerate(graph[i]):
            if j not in checked:
                new_w = table[i] + w #w = graph[i][j]. сумма весов от вершины до другой вершины
                if new_w < table[j]:
                    relations[j] = i
                    table[j] = new_w

        i = min_index(table, checked)
        if i >= 0:
            checked.add(i)

        #print(table, relations, checked, sep="\t")

    shortest_route = [end]

    while end != start:
        end = relations[shortest_route[-1]]
        shortest_route.append(end)

    return shortest_route[::-1]

#heap version
def dijkstra_h(graph: list[int], start: int, end: int) -> list:
    dist = {}
    rel = [0] * len(graph)
    h = [(0, start, start)]

    while h:
        d, n, pn = heapq.heappop(h) #takes the shortest available route
        if n in dist: continue
        
        dist[n] = d
        rel[n] = pn
        for sn in range(len(graph[n])):
            if sn in dist: continue
            heapq.heappush(h, (d + graph[n][sn], sn, n))

    route = []   
    curr = end
    while curr != start:
        route = [curr] + route
        curr = rel[curr]
    return [start] + route

#Prim
#find the shortest path to connect all points and create Minimum Spanning Tree
#the only difference from dijkstra is that we store graph[n][sn] instead of d + graph[n][sn]
def prim_h(graph: list[int]) -> int:
    total_d, mst = 0, []
    v = set()
    h = [(0, 0, 0)]
    while h:
        d, n, pn = heapq.heappop(h)
        if n in v: continue

        total_d += d
        mst.append((pn, n))
        v.add(n)
        for sn in range(len(graph[n])):
            if sn in v: continue
            heapq.heappush(h, (graph[n][sn], sn, n)) #so called difference
    return total_d, mst[1:]

#returns min distance without vertexes
def prim(graph: list[int]) -> int:
    total_d = 0
    dist = collections.defaultdict(lambda: float("inf"))
    v = set()
    n = 0
    for _ in range(len(graph) - 1):
        v.add(n)
        nxt = None
        for sn in range(len(graph)):
            if sn in v: continue
            dist[sn] = min(dist[sn], graph[n][sn])
            if nxt == None or dist[sn] < dist[nxt]: nxt = sn

        total_d += dist[nxt]
        n = nxt
    return total_d

#Floyd
#too hard in english
#выбирается вершина, через которую будут проходить пути ("вершина-посредник"),
#а потом для каждых вершин ищется путь через вершину-посредника
#если k - вершина-посредник, i -> j - путь, то: min(graph[i][j], graph[i][k] + graph[k][j) - пробитие пути (блатняк на хуторе хд)
def floyd(graph: list, start: int, end: int) -> list:
    graph_length = len(graph)

    table = [[to_v for to_v in range(graph_length)] for v in range(graph_length)]

    for k in range(graph_length): #k is basically a mediator vertex (вершина-посредник)
        for i in range(graph_length):
            for j in range(graph_length):
                d = graph[i][k] + graph[k][j] #e.g.: min(9, d(1,0) + d(0,2))
                if d < graph[i][j]:
                    graph[i][j] = d #update shortest path from i to j
                    table[i][j] = table[i][k] #change shortest relation vertex index to k 
                    #(NOTE: makes it so it only works from end to start, cause k -> i = j -> i (as directions))

    path = [end]
    while end != start:
        end = table[end][start] 
        path.append(end)

    return path[::-1]


