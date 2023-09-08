import heapq
import numpy as np


def dijkstra(graph, start, end):
    priority_queue = []
    heapq.heappush(priority_queue, [0, start])
    seen = set()
    path = []
    distance = np.inf
    while priority_queue:
        dist, node = heapq.heappop(priority_queue)
        path.append(node)
        if node == end:
            distance = dist
            break
        if node not in seen:
            seen.add(node)
            for node_ in graph[node]:
                if node_ not in seen:
                    heapq.heappush(priority_queue, [dist + graph[node][node_], node_])
    return path, distance


def bellman_ford(graph, start):
    prev_opt = np.ones(len(graph)) * np.inf
    prev_opt[start] = 0
    curr_opt = np.ones(len(graph)) * np.inf

    for i in range(len(graph)):
        for vertex in graph:
            curr_opt[vertex] = min([
                prev_opt[vertex],
                *[prev_opt[v] + graph[v][vertex] for v in graph if vertex in graph[v]]
            ])
        prev_opt = curr_opt[:]
    return curr_opt


def floyd_warshall(graph):
    distance = np.ones((len(graph), len(graph))) * np.inf
    for i in graph:
        distance[i, i] = 0
        for j in graph[i]:
            distance[i, j] = graph[i][j]

    for k in range(len(graph)):
        for i in range(len(graph)):
            for j in range(len(graph)):
                distance[i, j] = min(distance[i, j], distance[i, k] + distance[k, j])
    return distance


if __name__ == "__main__":
    graph = {
        0: {1: 5, 2: 1},
        1: {},
        2: {1: 1}
    }
    print(floyd_warshall(graph))
