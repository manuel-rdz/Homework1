from statistics import mean

from search import *

import matplotlib.pyplot as plt
import time


# function to benchmark different configurations of puzzle size, difficulty as well as number of times to run and
# algorithms used.
# It generates a plot with all the information of the benchmark
# parameters:
# title -> string title of the plot to generate
# searches -> set containing the code of the algorithms that we want to test
# size -> the size of the puzzle (size, size)
# difficulty -> represents the amount of moves to perform on the solved state to generate a unsolved puzzle config.
# if difficulty is set to -1, it will generate a completely solvable random configuration.
# if difficulty is set to -2, it will generate a random difficulty from (10 to 100).
# loop -> number of times to generate a new unsolved puzzle and add it to the benchmark.
def benchmark_puzzle(title, searches=None, size=3, difficulty=-1, loop=10):
    if searches is None:
        searches = {'ucs', 'mis', 'man', 'man_lm'}

    original_labels = ['ucs', 'A* misplaced', 'A* manhattan', 'A* manh_last_moves']
    times = [[], [], [], []]
    lengths = [[], [], [], []]
    random_difficulty = True if difficulty == -2 else False

    avg_solution_length = 0

    for i in range(loop):
        d = random.randint(10, 60) if random_difficulty else difficulty
        puzzle = PuzzleProblem(size, difficulty=d)

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

        print('Solved puzzle ', i)

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
    # random instances of 3x3 puzzle with all the algorithms 30 loop
    benchmark_puzzle(title='Average time solving 30 random instances of 3x3 puzzle',
                     size=3, difficulty=-1, loop=30)
    # random instances of 3x3 puzzle just with manhattan heuristics
    benchmark_puzzle(searches={'man', 'man_lm'}, title='Average time solving 100 random instances of 3x3 puzzle',
                     size=3, difficulty=-1, loop=100)

    # defined difficulty instances of 3x3-5x5 with all the algorithms
    for i in range(3, 7):
        benchmark_puzzle('Average time solving 30 random difficulty instances of ' + str(i) + 'x' + str(i) + ' puzzle',
                         size=i, difficulty=-2, loop=30)

    # defined difficulty instances of 3x3-5x5 with manhattan heuristics
    for i in range(6, 7):
        benchmark_puzzle(searches={'mis', 'man', 'man_lm'},
                         title='Average time solving 50 random difficulty instances of ' + str(i) + 'x' + str(i) + ' puzzle',
                         size=i, difficulty=-2, loop=50)