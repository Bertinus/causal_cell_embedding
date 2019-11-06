"""
This file contains generators of observed subgraphs which are suitable to represent links between hidden variables
and observations. Given a nxgraph, observation nodes are added and linked to hidden variables
"""

import numpy as np
import random


def random_obs_subgraph_generator(graph, n_observations, proba):
    """
    :param graph: nxgraph to which observation nodes will be added
    :param n_observations:
    :param proba: Proba to draw an edge between a parent and a child
    :return: bipartite graph where each observation has at least one parent. parent['bipartite'] == 0
    """
    hiddens = {n for n, d in graph.nodes(data=True)}  # set of hidden nodes
    n_hidden = len(graph)

    for idx in range(n_hidden, n_hidden + n_observations):
        graph.add_node(idx)  # Create node
        for parent in hiddens:  # Add parents
            if np.random.binomial(1, proba) == 1:
                graph.add_edge(parent, idx)
        # If no parent has been added, add one at random
        if graph.degree[idx] == 0:
            parent = random.sample(hiddens, 1)[0]
            graph.add_edge(parent, idx)
