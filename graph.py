from math import inf


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

        i = min_index(table,  checked)
        if i >= 0:
            checked.add(i)

        print(table, relations, checked, sep="\t")

    shortest_route = [end]

    while end != start:
        end = relations[shortest_route[-1]]
        shortest_route.append(end)

    return shortest_route[::-1]

#Floyd (!!!ОСУЖДАЕМ!!!George Droyd!!!ОСУЖДАЕМ!!!)
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


