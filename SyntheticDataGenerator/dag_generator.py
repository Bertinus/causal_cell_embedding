"""
This file contains several directed acyclic graph generators that are used to represent causal links between
hidden variables
"""
import networkx as nx
import numpy as np


def empty_graph_generator(n):
    return nx.empty_graph(n)


def gn_graph_generator(n, seed=0):
    return nx.gn_graph(n, seed=seed).reverse(copy=False)


def multi_gn_graph_generator(n, max_parents=3):
    g = nx.DiGraph()
    g.add_node(0)
    for i in range(1, n):
        g.add_node(i)
        n_parents = min(i, np.random.randint(0, max_parents+1))  # Sample number of parents
        parents = np.random.choice(range(i), size=n_parents, replace=False)  # Sample parents
        parent_edge_list = [(p, i) for p in parents]
        g.add_edges_from(parent_edge_list)
    return g


def hierarchical_generator(n):
    g = nx.DiGraph()
    g.add_node(0)
    for i in range(1, n):
        g.add_node(i)
        g.add_edges_from([(i-1, i)])
    return g
