import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from SyntheticDataGenerator.structural_equation import StructuralEquation, binary_lin_generator, Const, NoisyLinear, \
    binary_noisy_lin_generator
from SyntheticDataGenerator.dag_generator import gn_graph_generator, multi_gn_graph_generator
from SyntheticDataGenerator.obs_subgraph_generator import random_obs_subgraph_generator
from SyntheticDataGenerator.utils import circular_plus_obs_layout
import copy
import pickle


########################################################################################################################
# StructuralGraph class
########################################################################################################################


class StructuredGraph:
    """
    Object which contains the latent DAG, as well as observable nodes, and which can generate data wrt a set of
    structural equations
    """
    def __init__(self, n_hidden, n_observations,
                 directed_acyclic_graph_generator=gn_graph_generator,
                 obs_subgraph_generator=lambda g, n_obs: random_obs_subgraph_generator(g, n_obs, proba=0.2),
                 structural_equation_generator=binary_lin_generator, **kwargs):
        """
        :param n_hidden: Number of hidden nodes
        :param n_observations: Number of observed nodes
        :param directed_acyclic_graph_generator: Must return a directed acyclic graph networkx object.
        Takes number of hidden nodes as input
        Hand coded generator or one of the following :
        https://networkx.github.io/documentation/latest/reference/generators.html#module-networkx.generators.directed
        :param obs_subgraph_generator: Given a nxgraph and number of observation nodes, add nodes and edges
        corresponding to the observed part graph
        :param structural_equation_generator: given a graph and index, returns a structural
        equation and sets related weights in the graph.
        """
        self.n_hidden = n_hidden
        self.n_observations = n_observations
        self.n_nodes = n_hidden + n_observations

        # Initialize graph
        self.graph = nx.DiGraph(directed_acyclic_graph_generator(n_hidden))  # Make a copy to unfreeze the graph
        self.add_observation_nodes(obs_subgraph_generator)
        self.n_edges = self.graph.number_of_edges()
        self.initialize_graph(structural_equation_generator, **kwargs)

        # Used for interventions
        self.intervened_node = None
        self.truncated_edges = None
        self.intervened_structeq = None

    def add_observation_nodes(self, obs_subgraph_generator):
        """
        :param obs_subgraph_generator:
        :return:
        """
        obs_subgraph_generator(self.graph, self.n_observations)
        nx.set_node_attributes(self.graph, dict(zip(range(self.n_nodes),
                                                    [0] * self.n_hidden + [1] * self.n_observations)), 'observation')

    def initialize_graph(self, structural_equation_generator, **kwargs):
        """
        Initialize the graph self.graph by adding 'value' (int) attribute as well as 'structeq' (StructuralEquation)
        attribute  to each node. Also adds 'weight' attribute to each edge

        :param structural_equation_generator: given a graph and index, returns a structural
        equation and sets related weights in the graph.
        """
        nx.set_node_attributes(self.graph, dict(zip(range(self.n_nodes), [np.zeros(1)] * self.n_nodes)), 'value')
        # Initialize edge weights to one
        nx.set_edge_attributes(self.graph, dict(zip(self.graph.edges, [1] * self.n_edges)), name='weight')
        # Add struct eq to the graph object. Edge weights are (possibly) modified by the structural equation generator
        structeq_list = [structural_equation_generator(self.graph, i, **kwargs) for i in range(self.n_nodes)]
        nx.set_node_attributes(self.graph, dict(zip(range(self.n_nodes), structeq_list)), 'structeq')

    def get_observation_nodes(self):
        return range(self.n_hidden, self.n_nodes)

    def get_hidden_nodes(self):
        return range(self.n_hidden)

    def generate(self, n_examples=1):
        """
        Generates all values from structural equations in the right (topological) order.
        Note : Generated values are not dependent on the previous state of the graph as long as the graph is acyclic
        """
        for n in nx.topological_sort(self.graph):
            self.graph.nodes[n]["structeq"].generate(n_examples)

    def get_observations(self, with_hidden=False):
        """
        :return: An array of shape (n_examples, n_observations) filled with observation values where n_examples is
        the current shape of the "value" node attribute
        """
        data = np.array(list(nx.get_node_attributes(self.graph, 'value').values()))[self.n_hidden:].T
        if with_hidden:
            hidden = np.array(list(nx.get_node_attributes(self.graph, 'value').values()))[:self.n_hidden].T
            return hidden, data
        return data

    def set_intervention(self, node, function=Const(offset=0)):
        """
        Intervene on the graph in a hard manner, removing all incoming edges to the intervened node
        """
        # End previous intervention
        self.reset_intervention()
        # Save truncated part
        self.intervened_node = node
        self.truncated_edges = copy.deepcopy(self.graph.in_edges(node, data=True))
        self.intervened_structeq = self.graph.nodes[node]["structeq"]
        # Truncate the graph
        self.graph.remove_edges_from(self.truncated_edges)
        self.graph.nodes[node]["structeq"] = StructuralEquation(self.graph, node, function)

    def reset_intervention(self):
        # Recover original graph if the graph has been intervened
        if self.intervened_node:
            if self.truncated_edges:
                self.graph.add_edges_from(self.truncated_edges)
            self.graph.nodes[self.intervened_node]["structeq"] = self.intervened_structeq
            self.intervened_node = None
            self.truncated_edges = None
            self.intervened_structeq = None

    def set_soft_intervention(self, node,
                              function=NoisyLinear(lambda size: np.random.normal(loc=10.0, scale=1.0, size=size))):
        """
        Intervene on the graph in a soft manner, without removing incoming edges to the intervened node
        """
        self.reset_intervention()
        self.intervened_node = node
        self.intervened_structeq = self.graph.nodes[node]["structeq"]
        self.graph.nodes[node]["structeq"] = StructuralEquation(self.graph, node, function)

    def __str__(self):
        string = "DirectedAcyclicGraph. "
        string += str(len(self.graph)) + " nodes. " + str(self.graph.number_of_edges()) + " edges. Intervened node :" \
                  + str(self.intervened_node) + "\n"
        string += "Nodes :\n"
        for i in range(len(self.graph)):
            string += "\t node " + str(i) + " " + str(self.graph.nodes[i]) + "\n"
        string += "Edges :\n"
        for i in self.graph.edges.data():
            string += "\t edge " + str(i) + "\n"
        return string

    def draw(self, layout=circular_plus_obs_layout, show_node_name=False, show_values=False, show_eq=False,
             show_weights=False, colorbar=False, ax=None):
        """
        Draw the graph with matplotlib

        :param layout: given a graph, returns positions of nodes
        :param show_node_name:
        :param show_values: show values on top of nodes
        :param show_eq: Show struc. equation on top of edges
        :param show_weights:
        :param colorbar:
        :param ax: Matplotlib Axes object
        """
        pos = layout(self)
        # Print only first example of values for each node
        values = [i[0] for i in nx.get_node_attributes(self.graph, 'value').values()]

        nc = nx.draw_networkx_nodes(self.graph, pos, node_color=values, cmap=plt.cm.autumn, ax=ax)
        nx.draw_networkx_edges(self.graph, pos,
                               width=[2 * w for w in list(nx.get_edge_attributes(self.graph, "weight").values())],
                               ax=ax)
        if colorbar:
            plt.colorbar(nc)
        if show_node_name:
            # nx.draw_networkx_labels(self.graph, pos)
            nx.draw_networkx_labels(self.graph, pos=dict(zip(pos.keys(),
                                                             [x + np.array([0, 0.15]) for x in pos.values()])), ax=ax)
        if show_values:
            truncated_values = dict(zip(nx.get_node_attributes(self.graph, 'value').keys(),
                                        [str(x[0])[:5] for x in nx.get_node_attributes(self.graph, 'value').values()]))
            nx.draw_networkx_labels(self.graph, pos, truncated_values, ax=ax)
        if show_eq:
            nx.draw_networkx_labels(self.graph, pos=dict(zip(pos.keys(),
                                                             [x + np.array([0, -0.15]) for x in pos.values()])),
                                    labels=nx.get_node_attributes(self.graph, 'structeq'), ax=ax)
        if show_weights:
            truncated_weights = dict(zip(nx.get_edge_attributes(self.graph, 'weight').keys(),
                                         [str(x)[:5] for x in nx.get_edge_attributes(self.graph, 'weight').values()]))
            nx.draw_networkx_edge_labels(self.graph, pos, truncated_weights, ax=ax)
        if ax:
            ax.set_axis_off()
        else:
            plt.axis('off')

    def to_default_state(self):
        self.reset_intervention()
        self.generate(1)  # To not store lots of generated data


########################################################################################################################
# Tests
########################################################################################################################

if __name__ == "__main__":
    # G = StructuredGraph(4, 3, structural_equation_generator=binary_noisy_lin_generator, mean=0.0, var=0.01)
    G = StructuredGraph(10, 0, structural_equation_generator=binary_noisy_lin_generator,
                        directed_acyclic_graph_generator=multi_gn_graph_generator,
                        mean=0.0, var=0.01)
    np.random.seed(1)
    G.generate()
    G.draw(show_node_name=True, show_values=False, show_eq=False, show_weights=True, colorbar=False)
    plt.show()
    G.set_soft_intervention(3)
    G.generate()
    G.draw(show_node_name=True, show_values=False, show_eq=False, show_weights=True, colorbar=False)
    plt.show()
    # G.set_soft_intervention(1)
    # G.generate()
    # G.draw(show_node_name=False, show_values=True, show_eq=False, show_weights=True, colorbar=False)
    # plt.show()
    # G.reset_intervention()
    # G.generate()
    # G.draw(show_node_name=False, show_values=True, show_eq=False, show_weights=True, colorbar=False)
    # plt.show()
