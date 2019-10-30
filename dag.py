import networkx as nx
import matplotlib.pyplot as plt
import random
from structural_functions import Max, Linear


class StructuralEquation:
    """
    Object that represents a given structural equation. Structural equations are used to compute the value of a node
    given its predecessors. One can pass the function used to compute the value of the target node. This function can
    be stochastic.
    """
    def __init__(self, dag, index, function=Max, **kwargs):
        """
        :param dag: directed acyclic graph as a nxgraph object
        :param index: index of the target node
        """
        self.dag = dag
        self.target = index
        self.function = function(**kwargs)

    def __str__(self):
        return str(self.function)

    def __repr__(self):
        return str(self.function)

    def generate(self):
        inputs = [self.dag.node[i]['value'] for i in self.dag.predecessors(self.target)]
        edge_weights = [n[2]["weight"] for n in self.dag.in_edges(nbunch=self.target, data=True)]
        self.dag.node[self.target]['value'] = self.function.compute(inputs=inputs, edge_weights=edge_weights)


class DirectedAcyclicGraph:
    """
    Object which contains the latent DAG, as well as the functions to generate the
    values of the nodes
    """

    def __init__(self, n_nodes,
                 directed_graph_generator=lambda x: nx.gn_graph(x, seed=0).reverse(copy=False),
                 function=Max, **kwargs):
        """
        :param n_nodes: Number of nodes
        :param directed_graph_generator: Hand coded generator or one of the following :
        https://networkx.github.io/documentation/latest/reference/generators.html#module-networkx.generators.directed
        """
        self.n_nodes = n_nodes
        self.dag = directed_graph_generator(n_nodes)
        self.function=function
        self.initialize_dag(**kwargs)

    def initialize_dag(self, **kwargs):
        nx.set_node_attributes(self.dag, dict(zip(range(self.n_nodes), [0] * self.n_nodes)), 'value')
        structeq_list = [StructuralEquation(self.dag, i, function=self.function, **kwargs) for i in range(self.n_nodes)]
        nx.set_node_attributes(self.dag, dict(zip(range(self.n_nodes), structeq_list)), 'structeq')
        nx.set_edge_attributes(self.dag, 1, name="weight")

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

    def draw(self, layout="circular", show_node_name=False, show_values=False, show_eq=False):
        """
        Draw the DAG with matplotlib
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
        nx.draw_networkx_edges(self.dag, pos)
        plt.colorbar(nc)
        plt.axis('off')
        if show_node_name:
            nx.draw_networkx_labels(self.dag, pos)
        if show_values:
            nx.draw_networkx_labels(self.dag, pos, nx.get_node_attributes(self.dag, 'value'))
        if show_eq:
            nx.draw_networkx_labels(self.dag, pos, nx.get_node_attributes(self.dag, 'structeq'))
        plt.show()


if __name__ == "__main__":
    G = DirectedAcyclicGraph(10, function=Max, offset=0)
    print(G)
    G.draw(show_node_name=False, show_values=True, show_eq=False)
    G.generate()
    G.draw(show_node_name=False, show_values=True, show_eq=False)
    G.generate()
    G.draw(show_node_name=False, show_values=True, show_eq=False)
