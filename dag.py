import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from structural_functions import StructuralEquation, Max, Linear, NoisyLinear


class DirectedAcyclicGraph:
    """
    Object which contains the latent DAG, as well as the functions to generate the
    values of the nodes
    """

    def __init__(self, n_nodes,
                 directed_graph_generator=lambda x: nx.gn_graph(x, seed=0).reverse(copy=False),
                 function=Max,
                 edge_weight_sampler=lambda x: np.random.normal(loc=0.0, scale=1.0, size=x), **kwargs):
        """
        :param n_nodes: Number of nodes
        :param directed_graph_generator: Hand coded generator or one of the following :
        https://networkx.github.io/documentation/latest/reference/generators.html#module-networkx.generators.directed
        :param function: StructuralFunction object
        """
        self.n_nodes = n_nodes
        self.dag = directed_graph_generator(n_nodes)
        self.function = function
        self.initialize_dag(edge_weight_sampler, **kwargs)

    def initialize_dag(self, edge_weight_sampler, **kwargs):
        nx.set_node_attributes(self.dag, dict(zip(range(self.n_nodes), [0] * self.n_nodes)), 'value')
        # Add structural equations to the graph object
        structeq_list = [StructuralEquation(self.dag, i, function=self.function, **kwargs) for i in range(self.n_nodes)]
        nx.set_node_attributes(self.dag, dict(zip(range(self.n_nodes), structeq_list)), 'structeq')
        # Initialize edge weights
        weights_dict = dict(zip(self.dag.edges, edge_weight_sampler(len(self.dag.edges))))
        nx.set_edge_attributes(self.dag, weights_dict, name="weight")

    def generate(self):
        """
        generate all values from struct eq in the right (topological) order
        """
        for n in nx.topological_sort(self.dag):
            self.dag.node[n]["structeq"].generate()

    def __str__(self):
        string = "DirectedAcyclicGraph. "
        string += str(len(self.dag)) + " nodes. " + str(self.dag.number_of_edges()) + " edges.\n"
        string += "Nodes :\n"
        for i in range(len(self.dag)):
            string += "\t node " + str(i) + " " + str(self.dag.node[i]) + "\n"
        string += "Edges :\n"
        for i in self.dag.edges.data():
            string += "\t edge " + str(i) + "\n"
        return string

    def draw(self, layout="circular", show_node_name=False, show_values=False, show_eq=False,
             show_weights=False, colorbar=False):
        """
        Draw the DAG with matplotlib
        :param show_weights:
        :param colorbar:
        :param show_node_name:
        :param show_eq: Show struc. equation on top of edges
        :param show_values: show values on top of nodes
        :param layout: "circular" or "spring"
        """
        if layout == "spring":
            random.seed(0)
            # positions for all nodes, chosen in a deterministic way
            pos = nx.spring_layout(self.dag,
                                   iterations=100, pos=dict(zip(range(self.n_nodes), [(random.random(), random.random())
                                                                                      for _ in range(self.n_nodes)])))
        else:
            pos = nx.circular_layout(self.dag)

        values = list(nx.get_node_attributes(self.dag, 'value').values())
        nc = nx.draw_networkx_nodes(self.dag, pos, node_color=values, cmap=plt.cm.autumn)
        nx.draw_networkx_edges(self.dag, pos,
                               width=[2*w for w in list(nx.get_edge_attributes(self.dag, "weight").values())])
        if colorbar:
            plt.colorbar(nc)
        plt.axis('off')
        if show_node_name:
            nx.draw_networkx_labels(self.dag, pos)
        if show_values:
            truncated_values = dict(zip(nx.get_node_attributes(self.dag, 'value').keys(),
                                        [str(x)[:5] for x in nx.get_node_attributes(self.dag, 'value').values()]))
            nx.draw_networkx_labels(self.dag, pos, truncated_values)
        if show_eq:
            nx.draw_networkx_labels(self.dag, pos, nx.get_node_attributes(self.dag, 'structeq'))
        if show_weights:
            truncated_weights = dict(zip(nx.get_edge_attributes(self.dag, 'weight').keys(),
                                         [str(x)[:5] for x in nx.get_edge_attributes(self.dag, 'weight').values()]))
            nx.draw_networkx_edge_labels(self.dag, pos, truncated_weights)
        plt.show()


if __name__ == "__main__":
    G = DirectedAcyclicGraph(5, function=NoisyLinear, offset=10)
    print(G)
    G.draw(show_node_name=False, show_values=True, show_eq=False, show_weights=True)
    np.random.seed(0)
    G.generate()
    G.draw(show_node_name=False, show_values=True, show_eq=False, show_weights=True)
    np.random.seed(0)
    G.generate()
    G.draw(show_node_name=False, show_values=True, show_eq=False, show_weights=True)
