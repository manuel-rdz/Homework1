from statistics import mean

from search import *

import matplotlib.pyplot as plt
import numpy as np


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


if __name__ == "__main__":
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

    romania = NavigationProblem(NavigationState('A', location=locations['A']),
                                NavigationState('B', location=locations['B']),
                                connections,
                                locations=locations)  # for A*, you will need to also provide the locations

    # print(romania.successors('B'))  # [('go to S', 140, 'S'), ('go to Z', 75, 'Z'), ('go to T', 118, 'T')]
    # TODO: apply UCS, Greedy search and A*, the heuristic being the Euclidean

    print("Romania Graph")
    print("Find path from Node " + romania.initial.city + " to Node " + romania.goal.city)

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