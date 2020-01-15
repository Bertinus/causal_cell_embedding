"""
This file contains several directed acyclic graph generators that are used to represent causal links between
hidden variables
"""
import networkx as nx


def gn_graph_generator(n, seed=0):
    return nx.gn_graph(n, seed=seed).reverse(copy=False)


# TODO : Add more graph generators
