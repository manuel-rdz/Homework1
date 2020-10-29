from statistics import mean

from search import *

import matplotlib.pyplot as plt
import time


def benchmark_puzzle(title, searches=None, size=3, difficulty=-1, loop=10):
    if searches is None:
        searches = {'ucs', 'mis', 'man', 'man_lm'}

    original_labels = ['ucs', 'A* misplaced', 'A* manhattan', 'A* manh_last_moves']
    times = [[], [], [], []]
    lengths = [[], [], [], []]

    avg_solution_length = 0

    for i in range(loop):
        puzzle = PuzzleProblem(size, difficulty=difficulty)

        if 'ucs' in searches:
            tic = time.perf_counter()
            solution, _, _ = graph_search(puzzle, PriorityQueue(ucs))
            toc = time.perf_counter()
            times[0].append(toc - tic)
            lengths[0].append(len(solution.getPath()))

        if 'mis' in searches:
            tic = time.perf_counter()
            solution, _, _ = graph_search(puzzle, PriorityQueue(f_misplaced_tiles))
            toc = time.perf_counter()
            times[1].append(toc - tic)
            lengths[1].append(len(solution.getPath()))

        if 'man' in searches:
            tic = time.perf_counter()
            solution, _, _ = graph_search(puzzle, PriorityQueue(f_manhattan_distance))
            toc = time.perf_counter()
            times[2].append(toc - tic)
            lengths[2].append(len(solution.getPath()))

        if 'man_lm' in searches:
            tic = time.perf_counter()
            solution, _, _ = graph_search(puzzle, PriorityQueue(f_experimental))
            toc = time.perf_counter()
            times[3].append(toc - tic)
            lengths[3].append(len(solution.getPath()))

        if lengths[2][-1] != lengths[3][-1]:
            puzzle.initial.print_state()
            print(lengths[2][-1])
            print(lengths[3][-1])

    solution_mismatch = False
    values = []
    labels = []
    all_lengths = []
    for i, v in enumerate(times):
        if len(v) > 0:
            values.append(mean(v))
            labels.append(original_labels[i])
            if len(all_lengths) == 0:
                all_lengths = lengths[i]
            elif all_lengths != lengths[i]:
                solution_mismatch = True

    if not solution_mismatch:
        print('Optimal solution found by all searches')
        print(all_lengths)
        avg_solution_length = int(mean(all_lengths))
    else:
        print('Solution length mismatch:')
        for i, v in enumerate(lengths):
            print(i, v)

    plt.figure()
    plt.ylabel('Seconds')
    plt.xlabel('Search Algorithm')
    plt.bar(labels, values)
    plt.title(title + '\n avg solution length=' + str(avg_solution_length))
    plt.show()

    print("Average solution length: ", avg_solution_length)


if __name__ == "__main__":

    #puzzle = PuzzleProblem(3)

    #puzzle.initial = PuzzleState(matrix=np.array([[8, 5, 2], [3, 0, 4], [6, 7, 1]]))

    #solution, _, _ = graph_search(puzzle, PriorityQueue(ucs))
    #print(len(solution.getPath()))
    #solution, _, _ = graph_search(puzzle, PriorityQueue(f_manhattan_distance))
    #print(len(solution.getPath()))
    # random instances of 3x3 puzzle with all the algorithms 30 loop
    #benchmark_puzzle(title='Average time solving 30 random instances of 3x3 puzzle',
    #                 size=3, difficulty=-1, loop=30)
    # random instances of 3x3 puzzle just with manhattan heuristics
    #benchmark_puzzle(searches={'man', 'man_lm'}, title='Average time solving 100 random instances of 3x3 puzzle',
    #                 size=3, difficulty=-1, loop=50)
    benchmark_puzzle(title="hola", searches={'man', 'man_lm'}, size=20, difficulty=70, loop=5)
    # defined difficulty instances of 3x3-5x5 with all the algorithms
    # defined difficulty instances of 3x3-5x5 with manhattan heuristics