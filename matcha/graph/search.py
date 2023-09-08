def breadth_first_search(graph, start=0):
    queue = [start]
    seen = {start}
    search_result = []
    while queue:
        curr = queue.pop(0)
        search_result.append(curr)
        for node in graph[curr]:
            if node in seen:
                continue
            seen.add(node)
            queue.append(node)
    return search_result


def deep_first_search(graph, start=0):
    stack = [start]
    seen = {start}
    search_result = []
    while stack:
        curr = stack.pop()
        search_result.append(curr)
        for node in graph[curr]:
            if node in seen:
                continue
            seen.add(node)
            stack.append(node)
    return search_result
