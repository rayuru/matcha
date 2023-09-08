import random


__all__ = [
    "merge_sort", "quick_sort", "insertion_sort", "quick_sort_inplace",
    "heap_sort", "selection_sort", "bubble_sort"
]


def merge_sort(input_list):

    if len(input_list) <= 1:
        return input_list

    mid = len(input_list) // 2

    left_list = merge_sort(input_list[: mid])
    right_list = merge_sort(input_list[mid:])

    i = j = 0
    sorted_list = []
    while i < len(left_list) and j < len(right_list):
        if left_list[i] < right_list[j]:
            sorted_list.append(left_list[i])
            i += 1
        else:
            sorted_list.append(right_list[j])
            j += 1

    if i < len(left_list):
        sorted_list.extend(left_list[i:])
    if j < len(right_list):
        sorted_list.extend(right_list[j:])

    return sorted_list


def quick_sort(input_list, randomize=False):
    if len(input_list) <= 1:
        return input_list

    if randomize:
        selected_idx = random.choice(range(len(input_list)))
    else:
        selected_idx = 0

    benchmark = input_list[selected_idx]
    left_list = []
    right_list = []
    benchmarks = []
    for i in input_list:
        if i < benchmark:
            left_list.append(i)
        elif i > benchmark:
            right_list.append(i)
        else:
            benchmarks.append(i)
    left_list = quick_sort(left_list, randomize=randomize)
    right_list = quick_sort(right_list, randomize=randomize)

    return left_list + benchmarks + right_list


def quick_sort_inplace(input_list, randomize=False, start=0, end=None):

    if end is None:
        end = len(input_list) - 1
    if end - start < 1:
        return input_list

    if randomize:
        selected_idx = random.choice(range(start, end + 1))
        input_list[start], input_list[selected_idx] = input_list[selected_idx], input_list[start]

    thres = start
    check = thres + 1
    while check <= end:
        if input_list[check] < input_list[thres]:
            for i in range(check, thres, -1):
                input_list[i], input_list[i - 1] = input_list[i - 1], input_list[i]
            thres += 1
        check += 1

    quick_sort_inplace(input_list, randomize=randomize, start=start, end=thres - 1)
    quick_sort_inplace(input_list, randomize=randomize, start=thres + 1, end=end)
    return input_list


def insertion_sort(input_list):

    for i in range(1, len(input_list)):
        for j in range(i - 1, -1, -1):
            if input_list[j] > input_list[j + 1]:
                input_list[j], input_list[j + 1] = input_list[j + 1], input_list[j]
            else:
                break
    return input_list


def _heapify(input_list, index):
    left = index * 2 + 1
    right = index * 2 + 2
    min_index = min(
        [index, left, right],
        key=lambda x: input_list[x] if x < len(input_list) else float("inf")
    )

    if index != min_index:
        input_list[index], input_list[min_index] = input_list[min_index], input_list[index]
        _heapify(input_list, min_index)


def _build_heap(input_list):
    i = len(input_list) // 2 - 1
    while i >= 0:
        _heapify(input_list, i)
        i -= 1
    return input_list


def _heap_pop(heap):
    heap[0], heap[-1] = heap[-1], heap[0]
    max_value = heap.pop()
    if len(heap) > 0:
        _heapify(heap, 0)
    return max_value


def heap_sort(input_list):
    heap = _build_heap(input_list)
    sorted_list = []
    while heap:
        sorted_list.append(_heap_pop(heap))
    return sorted_list


def selection_sort(input_list):
    for i in range(len(input_list) - 1):
        min_index = i
        for j in range(i + 1, len(input_list)):
            if input_list[j] < input_list[min_index]:
                min_index = j
        input_list[min_index], input_list[i] = input_list[i], input_list[min_index]
    return input_list


def bubble_sort(input_list):
    for i in range(len(input_list) - 1):
        for j in range(i + 1, len(input_list)):
            if input_list[j] < input_list[j - 1]:
                input_list[j], input_list[j - 1] = input_list[j - 1], input_list[j]
    return input_list
