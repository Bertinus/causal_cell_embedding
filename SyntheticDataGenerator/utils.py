"""
This file contains functions used for graph visualization
"""

import networkx as nx
import random
import numpy as np


########################################################################################################################
# Layouts
########################################################################################################################
"""
Returns position for each node in the graph as a 
"""


def circular_layout(stg):
    """
    :param stg: StructuredGraph object
    :return: dictionary where keys are node indices and values are 1D-arrays containing node coordinates
    """
    return nx.circular_layout(stg.graph)


def spring_layout(stg):
    random.seed(0)
    # positions for all nodes, chosen in a deterministic way
    return nx.spring_layout(stg.graph,
                            iterations=100,
                            pos=dict(zip(range(stg.n_nodes), [(random.random(), random.random())
                                                              for _ in range(stg.n_nodes)])))


def circular_plus_obs_layout(stg):
    pos = {}
    # Positions of hidden nodes
    for i in range(stg.n_hidden):
        x = 2 * np.pi * i / stg.n_hidden
        pos[i] = np.array([np.cos(x), np.sin(x)])
    # Positions of observation nodes
    for i in range(stg.n_hidden, stg.n_nodes):
        x = 4./(stg.n_observations-1) * (i - stg.n_hidden) - 2
        pos[i] = np.array([x, -2])

    return pos

