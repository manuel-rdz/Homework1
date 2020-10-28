from copy import deepcopy
from math import ceil

import networkx as nx
import numpy as np
import bisect
import random


class NavigationProblem:
    def __init__(self, initial, goal, connections, locations=None, directed=False):
        self.initial = initial
        self.initial.goal = goal
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
        return [("go to %s" % city,
                 connection['cost'],
                 NavigationState(city, self.locations[city], self.goal)) for city, connection in self.graph[state.city].items()]

    def goal_test(self, state):
        return state == self.goal


class NavigationState:
    def __init__(self, city, location, goal=None):
        self.city = city
        self.location = location
        self.goal = goal


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
    exploration_history = []
    while not frontier.empty():
        node = frontier.pop()
        if problem.goal_test(node.state):
            exploration_history.append(node)
            return node, exploration_history, frontier.max_frontier
        if node.state not in closed:
            exploration_history.append(node)
            closed.add(node.state)
            if limit_level is not None and node.level == limit_level:
                continue
            successors = node.expand(problem)
            for snode in successors:
                frontier.push(snode)

    return None, exploration_history, None


def graph_search_cycle_detection(problem, frontier, limit_level=None):
    frontier.push(Node(problem.initial, level=1))
    path = []
    exploration_history = []
    closed = set()
    while not frontier.empty():
        node = frontier.pop()
        while len(path) > 0 and node.level <= path[-1].level:
            closed.remove(path[-1].state)
            path.pop()
        if node.state in closed:
            continue
        path.append(node)
        exploration_history.append(node)
        closed.add(node.state)
        if problem.goal_test(node.state):
            return node, exploration_history, frontier.max_frontier
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
    return np.linalg.norm(np.asarray(node.goal.location) - np.asarray(node.state.location))


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


def manhattan_last_moves(state):
    distance = 0
    a = state.size * state.size - 1
    b = (state.size - 1) * state.size
    col_a, row_b = 0, 0
    for i in range(state.size):
        for j in range(state.size):
            v = state.matrix[i][j]
            if v == 0:
                distance += abs(i - (state.size - 1)) + abs(j - (state.size - 1))
            else:
                r = ceil(v / state.size) - 1
                c = v - r * state.size - 1
                distance += abs(i - r) + abs(j - c)
            if v == a:
                col_a = j
            if v == b:
                row_b = i
    if col_a != state.size - 1 and row_b != state.size - 1:
        distance += 2
    return distance


def f_euclidean(node):
    return node.path_cost + euclidean_distance(node)


def f_misplaced_tiles(node):
    return node.path_cost + misplaced_tiles(node.state)


def f_manhattan_distance(node):
    return node.path_cost + manhattan_distance(node.state)


def f_experimental(node):
    return node.path_cost + manhattan_last_moves(node.state)