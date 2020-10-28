from copy import deepcopy
from math import ceil

import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import bisect
import random
import argparse
from statistics import mean


class NavigationProblem:
    def __init__(self, initial, goal, connections, locations=None, directed=False):
        self.initial = initial
        self.goal = goal
        self.locations = locations
        self.graph = nx.DiGraph() if directed else nx.Graph()
        for cityA, cityB, distance in connections:
            if locations is not None:
                self.graph.add_node(cityA, pos=locations[cityA])
                self.graph.add_node(cityB, pos=locations[cityB])
            self.graph.add_edge(cityA, cityB, cost=distance)

    def successors(self, state):
        # Exactly as defined in Lecture slides,
        return [("go to %s" % city, connection['cost'], city) for city, connection in self.graph[state].items()]

    def goal_test(self, state):
        return state == self.goal


class PuzzleState:
    def __init__(self, matrix=None, goal=False, init=False, difficulty=-1, size=3):
        self.size = size
        if not matrix is None:  # 0 represents empty spot
            self.matrix = matrix
        else:
            permutation = np.append(np.arange(1, size * size), 0)
            if init:
                if difficulty == -1:
                    while True:
                        random.shuffle(permutation)
                        if self.__valid_permutation(permutation, size):
                            break
                else:
                    self.matrix = permutation.reshape((size, size))
                    for i in range(difficulty):
                        next_states = self.successors()
                        idx = random.randint(0, len(next_states) - 1)
                        self.matrix = next_states[idx][2].matrix
                    return
            self.matrix = permutation.reshape((size, size))

    def __valid_permutation(self, permutation, size):
        inversions = 0
        for i, v in enumerate(permutation):
            if v == 0:
                continue
            for j in range(i):
                if permutation[i] < permutation[j]:
                    inversions += 1
        if size & 1:  # odd size
            return not (inversions & 1)
        else:
            row = 0
            for idx, v in enumerate(permutation):
                if v == 0:
                    row = int(idx / size)
                    break
            row = size - row
            if row & 1:
                return not (inversions & 1)

            return inversions & 1

    def successors(self):
        actions_labels = ["left", "up", "right", "down"]
        actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        successors = []
        for i in range(self.size):
            for j in range(self.size):
                if self.matrix[i][j] == 0:
                    for idx, action in enumerate(actions):
                        ni = i + action[0]
                        nj = j + action[1]
                        if 0 <= ni < self.size and 0 <= nj < self.size:
                            self.matrix[i][j], self.matrix[ni][nj] = self.matrix[ni][nj], self.matrix[i][j]
                            successors.append(("move tile " + str(self.matrix[i][j]) + " " + actions_labels[idx], 1,
                                                PuzzleState(deepcopy(self.matrix), size=self.size)))
                            self.matrix[i][j], self.matrix[ni][nj] = self.matrix[ni][nj], self.matrix[i][j]
                    i = self.size
                    break
        return successors

    def is_goal(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.matrix[i][j] != self.size * i + j + 1:
                    if i != self.size - 1 or j != self.size - 1 or self.matrix[i][j] != 0:
                        return False
        return True

    def print_state(self):
        print(self.matrix)

    def __hash__(self):
        return hash(tuple(self.matrix.flatten()))

    def __eq__(self, other):
        return np.alltrue(other.matrix == self.matrix)


class PuzzleProblem:
    def __init__(self, size=3, difficulty=-1):  # size 3 means 3x3 field
        self.size = size
        self.initial = PuzzleState(init=True, size=size, difficulty=difficulty)  # init state is shuffled
        self.goal = PuzzleState(goal=True, size=size)  # goal state is unshuffled

    def successors(self, state):
        return state.successors()
        # Successors can be "outsourced", you can also calculate successor states here

    def goal_test(self, state):
        return np.alltrue(state.matrix == self.goal.matrix)


class Node:
    def __init__(self, state=None, parent=None, action=None, path_cost=0, level=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.level = level

    def getPath(self):
        """getting the path of parents up to the root"""
        currentNode = self
        path = [self]
        while currentNode.parent:  # stops when parent is None, ie root
            path.append(currentNode.parent)
            currentNode = currentNode.parent
        path.reverse()  # from root to this node
        return path

    def expand(self, problem):
        successors = problem.successors(self.state)
        return [Node(newState, self, action, self.path_cost + cost, 0 if self.level is None else self.level + 1) for
                (action, cost, newState) in
                successors]

    def __gt__(self, other):  # needed for tie breaks in priority queues
        return True

    def __repr__(self):
        return self.state, self.action, self.path_cost


class FIFO:
    def __init__(self):
        self.list = []
        self.max_frontier = 0

    def push(self, item):
        self.list.insert(0, item)
        self.max_frontier = max(len(self.list), self.max_frontier)

    def pop(self):
        return self.list.pop()

    def empty(self):
        return not len(self.list)


class LIFO:  # fill out yourself!
    def __init__(self):
        self.list = []
        self.max_frontier = 0

    def push(self, item):
        self.list.insert(0, item)
        self.max_frontier = max(len(self.list), self.max_frontier)

    def pop(self):
        return self.list.pop(0)

    def empty(self):
        return not len(self.list)


class PriorityQueue:
    def __init__(self, f):
        self.list = []
        self.f = f
        self.max_frontier = 0

    def push(self, item):
        priority = self.f(item)
        bisect.insort(self.list, (priority, random.random(), item))
        self.max_frontier = max(len(self.list), self.max_frontier)

    def pop(self):
        return self.list.pop(0)[-1]

    def empty(self):
        return not len(self.list)


def graph_search(problem, frontier, limit_level=None):
    """Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    If two paths reach a state, only use the best one. [Fig. 3.18]"""
    closed = set()  # sets can store hashable objects, thats why we need to define a hash code for states
    frontier.push(Node(problem.initial, level=0))
    explorationHistory = []
    while not frontier.empty():
        node = frontier.pop()
        if problem.goal_test(node.state):
            explorationHistory.append(node)
            return node, explorationHistory, frontier.max_frontier
        if node.state not in closed:
            explorationHistory.append(node)
            closed.add(node.state)
            if limit_level is not None and node.level == limit_level:
                continue
            successors = node.expand(problem)
            for snode in successors:
                frontier.push(snode)

    return None, explorationHistory, None


def graph_search_cycle_detection(problem, frontier, limit_level=None):
    frontier.push(Node(problem.initial, level=1))
    path = []
    explorationHistory = []
    closed = set()
    while not frontier.empty():
        node = frontier.pop()
        while len(path) > 0 and node.level <= path[-1].level:
            closed.remove(path[-1].state)
            path.pop()
        if node.state in closed:
            continue
        path.append(node)
        explorationHistory.append(node)
        closed.add(node.state)
        if problem.goal_test(node.state):
            return node, explorationHistory, frontier.max_frontier
        if limit_level is not None and node.level == limit_level:
            continue
        successors = node.expand(problem)
        for snode in successors:
            frontier.push(snode)

    return None, None, None


def iterative_deepening_graph_search(problem):
    i = 1
    while True:
        s, his, mf = graph_search_cycle_detection(problem, LIFO(), i)
        if s is not None:
            return s, his, mf
        i += 1


def breadth_first_graph_search(problem):
    return graph_search(problem, FIFO())


def depth_first_graph_search(problem):
    return graph_search(problem, LIFO())


def astar_graph_search(problem, f):
    return graph_search(problem, PriorityQueue(f))


# %%
# priority functions for Priority Queues used in UCS and A*, resp., if you are unfamiliar with lambda calc.

def ucs(node):
    return node.path_cost


def euclidean_distance(node):
    return np.linalg.norm(np.asarray(locations[romania.goal]) - np.asarray(locations[node.state]))


def misplaced_tiles(state):
    misplaced = 0
    for i in range(state.size):
        for j in range(state.size):
            if state.matrix[i][j] != (i * state.size) + j + 1:
                if i != state.size - 1 or j != state.size - 1 or state.matrix[i][j] != 0:
                    misplaced += 1
    return misplaced


def manhattan_distance(state):
    distance = 0
    for i in range(state.size):
        for j in range(state.size):
            v = state.matrix[i][j]
            if v == 0:
                distance += abs(i - (state.size - 1)) + abs(j - (state.size - 1))
            else:
                r = ceil(v / state.size) - 1
                c = v - r * state.size - 1
                distance += abs(i - r) + abs(j - c)
    return distance


def linear_conflicts(state):
    

def f_euclidean(node):
    return node.path_cost + euclidean_distance(node)


def f_misplaced_tiles(node):
    return node.path_cost + misplaced_tiles(node.state)


def f_manhattan_distance(node):
    return node.path_cost + manhattan_distance(node.state)


def experimental(state):
    pass


def f_experimental(node):
    return node.path_cost + experimental(node.state)


def plot_stats(title, xlabel, ylabel, x_labels, x, legend_labels, values):
    f, ax = plt.subplots(figsize=(20, 10))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x, x_labels)
    plt.yticks(np.arange(0, 20.1, 1))

    for i, h in enumerate(values):
        ax.bar(x + position[i], h, width=0.2, color=colors[i], align='center', label=legend_labels[i])

    ax.autoscale(tight=True)
    ax.legend()
    plt.show()


def update_lists(idx, nodes_expanded, solution_length, path_cost, max_frontier):
    exploration_history_sizes[idx].append(nodes_expanded)
    solution_sizes[idx].append(solution_length)
    path_costs[idx].append(path_cost / 100)
    max_frontiers[idx].append(max_frontier)


def search_algorithms_main():
    global exploration_history_sizes
    global solution_sizes
    global path_costs
    global max_frontiers
    global avg_statistics
    global position
    global colors
    global locations
    global romania

    # Romania
    connections = [('A', 'S', 140), ('A', 'Z', 75), ('A', 'T', 118), ('C', 'P', 138), ('C', 'R', 146), ('C', 'D', 120),
                   ('B', 'P', 101),
                   ('B', 'U', 85), ('B', 'G', 90), ('B', 'F', 211), ('E', 'H', 86), ('D', 'M', 75), ('F', 'S', 99),
                   ('I', 'V', 92),
                   ('I', 'N', 87), ('H', 'U', 98), ('L', 'M', 70), ('L', 'T', 111), ('O', 'S', 151), ('O', 'Z', 71),
                   ('P', 'R', 97), ('R', 'S', 80), ('U', 'V', 142)]

    locations = {'A': (91, 492), 'C': (253, 288), 'B': (400, 327), 'E': (562, 293), 'D': (165, 299), 'G': (375, 270),
                 'F': (305, 449),
                 'I': (473, 506), 'H': (534, 350), 'M': (168, 339), 'L': (165, 379), 'O': (131, 571), 'N': (406, 537),
                 'P': (320, 368),
                 'S': (207, 457), 'R': (233, 410), 'U': (456, 350), 'T': (94, 410), 'V': (509, 444), 'Z': (108, 531)}

    romania = NavigationProblem('A', 'B', connections,
                                locations=locations)  # for A*, you will need to also provide the locations

    # print(romania.successors('B'))  # [('go to S', 140, 'S'), ('go to Z', 75, 'Z'), ('go to T', 118, 'T')]
    # TODO: apply UCS, Greedy search and A*, the heuristic being the Euclidean

    print("Romania Graph")
    print("Find path from Node " + romania.initial + " to Node " + romania.goal)

    # DFS
    solution, history, _ = depth_first_graph_search(romania)
    print("DFS Solution:", [(node.state, node.action) for node in solution.getPath()])
    print("exploration history:", [node.state for node in history])

    # BFS
    solution, history, _ = breadth_first_graph_search(romania)
    print("BFS Solution:", [(node.state, node.action) for node in solution.getPath()])
    print("exploration history:", [node.state for node in history])

    # Iterative Deepening
    solution, history, _ = iterative_deepening_graph_search(romania)
    print("Iterative Deepening Solution:", [(node.state, node.action) for node in solution.getPath()])
    print("exploration history:", [node.state for node in history])

    # Uniform cost:
    solution, history, _ = graph_search(romania, PriorityQueue(ucs))
    print("UCS Solution:", [(node.state, node.action) for node in solution.getPath()])
    print("exploration history:", [node.state for node in history])

    # Best first
    solution, history, _ = graph_search(romania, PriorityQueue(euclidean_distance))
    print("Greedy Search Solution:", [(node.state, node.action) for node in solution.getPath()])
    print("exploration history:", [node.state for node in history])

    # A*
    solution, history, _ = graph_search(romania, PriorityQueue(f_euclidean))
    print("A* Solution:", [(node.state, node.action) for node in solution.getPath()])
    print("exploration history:", [node.state for node in history])
    print("Path cost:", solution.path_cost)

    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    position = [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]
    exploration_history_sizes = []
    solution_sizes = []
    path_costs = []
    max_frontiers = []
    avg_statistics = []
    search_algorithms_labels = ["DFS", "BFS", "ID", "UCS", "BF", "A*"]
    x_labels_stats = ["Nodes expansion", "Solution length", "Path cost (10^-2)", "Max frontier"]
    x_labels = [k for k in locations]
    x = np.arange(len(x_labels)) * 1.5
    for i in range(len(search_algorithms_labels)):
        exploration_history_sizes.append([])
        solution_sizes.append([])
        path_costs.append([])
        max_frontiers.append([])
        avg_statistics.append([])

    for s in locations:
        romania = NavigationProblem(s, 'B', connections, locations=locations)

        # DFS
        solution, history, max_frontier = depth_first_graph_search(romania)
        update_lists(0, len(history), len(solution.getPath()), solution.path_cost, max_frontier)
        # BFS
        solution, history, max_frontier = breadth_first_graph_search(romania)
        update_lists(1, len(history), len(solution.getPath()), solution.path_cost, max_frontier)
        # Iterative Deepening
        solution, history, max_frontier = iterative_deepening_graph_search(romania)
        update_lists(2, len(history), len(solution.getPath()), solution.path_cost, max_frontier)
        # UCS
        solution, history, max_frontier = graph_search(romania, PriorityQueue(ucs))
        update_lists(3, len(history), len(solution.getPath()), solution.path_cost, max_frontier)
        # Best First
        solution, history, max_frontier = graph_search(romania, PriorityQueue(euclidean_distance))
        update_lists(4, len(history), len(solution.getPath()), solution.path_cost, max_frontier)
        # A*
        solution, history, max_frontier = graph_search(romania, PriorityQueue(f_euclidean))
        update_lists(5, len(history), len(solution.getPath()), solution.path_cost, max_frontier)

    for i in range(len(search_algorithms_labels)):
        avg_statistics[i].append(mean(exploration_history_sizes[i]))
        avg_statistics[i].append(mean(solution_sizes[i]))
        avg_statistics[i].append(mean(path_costs[i]))
        avg_statistics[i].append(mean(max_frontiers[i]))

    # Nodes expansion
    plot_stats("Nodes expanded by each search algorithm", "Starting state", "Nodes expanded", x_labels, x,
               search_algorithms_labels, exploration_history_sizes)
    # Solution length
    plot_stats("Solutions length of each search algorithm", "Starting state", "Solution length", x_labels, x,
               search_algorithms_labels, solution_sizes)
    # Path cost
    plot_stats("Path cost of solution by each search algorithm", "Starting state", "Solution cost (10^-2)", x_labels, x,
               search_algorithms_labels, path_costs)
    # Max frontier
    plot_stats("Max frontier stored by each algorithm", "Starting state", "Max frontier size", x_labels, x,
               search_algorithms_labels, max_frontiers)
    # Avg statistics
    plot_stats("Avg statistics for each search algorithm", "Statistic", "Average", x_labels_stats,
               np.arange(len(x_labels_stats)) * 1.4, search_algorithms_labels, avg_statistics)

    import matplotlib.pylab as pl

    pl.clf()
    G = romania.graph
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True)
    labels = nx.get_edge_attributes(G, 'cost')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    pl.show()


def puzzle_main():
    puzzle = PuzzleProblem(3)
    # print(misplaced_tiles(puzzle.initial))
    # puzzle.initial.print_state()

    ucs_s = []
    astar_h1 = []
    astar_h2 = []

    #tic = time.perf_counter()
    #solution, _, _ = graph_search(puzzle, PriorityQueue(f_experimental))
    #toc = time.perf_counter()
    #print(f"A* mine took {toc - tic:0.4f} seconds")
    tic = time.perf_counter()
    solution, _, _ = graph_search(puzzle, PriorityQueue(f_manhattan_distance))
    toc = time.perf_counter()
    for node in solution.getPath():
        astar_h2.append(node.action)
    print(f"A* manhattan took {toc - tic:0.4f} seconds")
    tic = time.perf_counter()
    solution, _, _ = graph_search(puzzle, PriorityQueue(f_misplaced_tiles))
    toc = time.perf_counter()
    for node in solution.getPath():
        astar_h1.append(node.action)
    print(f"A* misplaced took {toc - tic:0.4f} seconds")
    tic = time.perf_counter()
    solution, _, _ = graph_search(puzzle, PriorityQueue(ucs))
    toc = time.perf_counter()
    for node in solution.getPath():
        ucs_s.append(node.action)
    print(f"UCS took {toc - tic:0.4f} seconds")
    print(len(ucs_s), len(astar_h1), len(astar_h2))
    # print('exploration matrix')
    # for node in history:
    #    print(node.state.matrix, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--puzzle", help="Run the n-puzzle search problem", action="store_true")
    parser.add_argument("-s", "--search", help="Run the different search algorithms on the Romania graph",
                        action="store_true")

    args = parser.parse_args()
    if args.puzzle:
        puzzle_main()
    elif args.search:
        search_algorithms_main()
    else:
        print("You must pass one of the two available flags for running this program. "
              "Refer to --help for additional info.")
