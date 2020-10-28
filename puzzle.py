from statistics import mean

from search import *

import matplotlib.pyplot as plt
import time


def benchmark_puzzle(searchs=None, size=3, difficulty=-1, loop=10):
    if searchs is None:
        searchs = {'ucs', 'mis', 'man', 'man_lm'}

    original_labels = ['ucs', 'misplaced', 'manhattan', 'manhattan_last_moves']
    times = [[], [], [], []]

    for i in range(loop):
        puzzle = PuzzleProblem(size, difficulty=difficulty)

        if 'ucs' in searchs:
            tic = time.perf_counter()
            solution, _, _ = graph_search(puzzle, PriorityQueue(ucs))
            toc = time.perf_counter()
            times[0].append(toc - tic)

        if 'mis' in searchs:
            tic = time.perf_counter()
            solution, _, _ = graph_search(puzzle, PriorityQueue(f_misplaced_tiles))
            toc = time.perf_counter()
            times[1].append(toc - tic)

        if 'man' in searchs:
            tic = time.perf_counter()
            solution, _, _ = graph_search(puzzle, PriorityQueue(f_manhattan_distance))
            toc = time.perf_counter()
            times[2].append(toc - tic)

        if 'man_lm' in searchs:
            tic = time.perf_counter()
            _, _, _ = graph_search(puzzle, PriorityQueue(f_experimental))
            toc = time.perf_counter()
            times[3].append(toc - tic)

    values = []
    labels = []
    for i, v in enumerate(times):
        if len(v) > 0:
            values.append(mean(v))
            labels.append(original_labels[i])

    plt.figure()
    plt.bar(labels, values)
    plt.show()


if __name__ == "__main__":
    benchmark_puzzle(searchs={'man', 'man_lm'}, size=3, difficulty=-1, loop=30)