"""
This file contains the StructuralEquation class as well as the functions that are possible to use
in structural equations
"""
import numpy as np
import networkx as nx

########################################################################################################################
# Structural Equation class
########################################################################################################################


class StructuralEquation:
    """
    Object that represents a given structural equation. Structural equations are used to compute the value of a node
    given its predecessors. One can pass the function used to compute the value of the target node. This function can
    be stochastic.
    """

    def __init__(self, graph, index, function):
        """
        :param graph: (directed acyclic) graph as a nxgraph object
        :param index: index of the target node
        :param function: StructuralFunction object.
        """
        self.graph = graph
        self.target = index
        self.function = function

    def __str__(self):
        return str(self.function)

    def __repr__(self):
        return str(self.function)

    def generate(self, n_examples=1):
        inputs = np.array([self.graph.node[i]['value'] for i in self.graph.predecessors(self.target)])
        edge_weights = np.array([n[2]["weight"] for n in self.graph.in_edges(nbunch=self.target, data=True)])
        self.graph.node[self.target]['value'] = self.function.compute(inputs, edge_weights, n_examples)


########################################################################################################################
# Structural function classes
########################################################################################################################


class StructuralFunction:
    def __init__(self, **kwargs):
        pass

    def compute(self, inputs, edge_weights=None, n_examples=1):
        """
        :param inputs: list of neighbours values
        :param edge_weights: list of edge weights
        :param n_examples:
        :return: output of the function
        """
        pass


class Max(StructuralFunction):
    def __init__(self, offset=0):
        super().__init__()
        self.offset = offset
        return

    def __str__(self):
        return "max"

    def __repr__(self):
        return "max"

    def compute(self, inputs, edge_weights=None, n_examples=1):
        if inputs.shape[0] == 0:
            return np.array([self.offset]*n_examples)
        return inputs.max(axis=0) + np.array([self.offset]*n_examples)


class Const(StructuralFunction):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset
        return

    def __str__(self):
        return "const"

    def __repr__(self):
        return "const"

    def compute(self, inputs, edge_weights=None, n_examples=1):
        return np.array([self.offset]*n_examples)


class Linear(StructuralFunction):
    def __init__(self, offset=0):
        super().__init__()
        self.offset = offset
        return

    def __str__(self):
        return "linear"

    def __repr__(self):
        return "linear"

    def compute(self, inputs, edge_weights=None, n_examples=1):
        if edge_weights.shape[0] == 0:
            return np.array([self.offset]*n_examples)
        return edge_weights[None, :].dot(inputs)[0, :] + np.array([self.offset]*n_examples)


class Noise(StructuralFunction):
    def __init__(self, sampler=lambda size: np.random.normal(loc=0.0, scale=1.0, size=size), offset=0):
        super().__init__()
        self.sampler = sampler
        self.offset = offset
        return

    def __str__(self):
        return "noise"

    def __repr__(self):
        return "noise"

    def compute(self, inputs, edge_weights=None, n_examples=1):
        return self.sampler(n_examples) + np.array([self.offset]*n_examples)


class NoisyLinear(StructuralFunction):
    def __init__(self, sampler=lambda size: np.random.normal(loc=0.0, scale=1.0, size=size), offset=0):
        super().__init__()
        self.sampler = sampler
        self.offset = offset
        return

    def __str__(self):
        return "noisy linear"

    def __repr__(self):
        return "noisy linear"

    def compute(self, inputs, edge_weights=None, n_examples=1):
        if edge_weights.shape[0] == 0:
            return self.sampler(n_examples) + np.array([self.offset]*n_examples)
        return edge_weights[None, :].dot(inputs)[0, :] + np.array([self.offset]*n_examples) + self.sampler(n_examples)


# TODO : Add quadratic and sinusoid functions


########################################################################################################################
# Structural Equation Generators
########################################################################################################################
"""
Those functions generate structural equations given a graph and an index. It allows to generate different types of 
equations for different types of nodes in the graph (e.g. hidden vs observed). 
Also sets edge weights to desired values
"""


def max_generator(graph, index):
    """
    :param graph: (directed acyclic) graph as a nxgraph object
    :param index: index of the target node
    """
    function = Max()
    return StructuralEquation(graph, index, function)


def lin_generator(graph, index):
    """
    :param graph: (directed acyclic) graph as a nxgraph object
    :param index: index of the target node
    """
    def edge_weight_sampler(g, edges):
        return np.random.normal(loc=0.0, scale=1.0, size=len(edges))

    function = NoisyLinear()
    set_weights(graph, graph.in_edges(index), edge_weight_sampler)

    return StructuralEquation(graph, index, function)


def binary_lin_generator(graph, index):
    """
    :param graph: (directed acyclic) graph as a nxgraph object
    :param index: index of the target node
    """
    def edge_weight_sampler(g, edges):
        return 2*np.random.binomial(1, 0.5, size=len(edges)) - 1

    function = Linear()
    set_weights(graph, graph.in_edges(index), edge_weight_sampler)

    # If a node has no parent, generate noise
    if len(list(graph.predecessors(index))) == 0:
        function = Noise()

    return StructuralEquation(graph, index, function)


def binary_noisy_lin_generator(graph, index):
    """
    :param graph: (directed acyclic) graph as a nxgraph object
    :param index: index of the target node
    """
    def edge_weight_sampler(g, edges):
        return 2*np.random.binomial(1, 0.5, size=len(edges)) - 1

    function = NoisyLinear()
    set_weights(graph, graph.in_edges(index), edge_weight_sampler)

    return StructuralEquation(graph, index, function)


def lin_hidden_max_obs_generator(graph, index):
    """
    Linear equations for hidden nodes and max for observed ones
    :param graph: (directed acyclic) graph as a nxgraph object
    :param index: index of the target node
    """
    def edge_weight_sampler(g, edges):
        return np.random.normal(loc=0.0, scale=1.0, size=len(edges))

    if graph.node[index]['observation'] == 0:
        set_weights(graph, graph.in_edges(index), edge_weight_sampler)
        return StructuralEquation(graph, index, function=NoisyLinear())

    return StructuralEquation(graph, index, function=Max())


########################################################################################################################
# Other functions
########################################################################################################################

def set_weights(graph, edges, edge_weight_sampler):
    weights_dict = dict(zip(edges, edge_weight_sampler(graph, edges)))
    nx.set_edge_attributes(graph, weights_dict, name='weight')


if __name__ == "__main__":
    noisy_lin = NoisyLinear(offset=10)
    print(type(noisy_lin.compute(0)))
    print(noisy_lin.compute(0))
    print(noisy_lin.compute(0))
