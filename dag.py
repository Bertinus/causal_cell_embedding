import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import pdb


class DirectedAcyclicGraph:
    """
    Object which contains the latent DAG, as well as the functions to generate the
    values of the nodes
    """

    def __init__(self, n_nodes,
                 directed_graph_generator=lambda x: nx.gn_graph(x, seed=0).reverse(copy=False)):
        """
        :param n_nodes: Number of nodes
        :param directed_graph_generator: Hand coded generator or one of the following :
        https://networkx.github.io/documentation/latest/reference/generators.html#module-networkx.generators.directed
        """
        self.n_nodes = n_nodes
        self.dag = directed_graph_generator(n_nodes)
        nx.set_node_attributes(self.dag, dict(zip(range(self.n_nodes), [0] * self.n_nodes)), 'value')
        self.functions = dict(zip(range(self.n_nodes), [self.function_generator(self.dag.predecessors(i),
                                                                                self.dag.node(i))
                                                        for i in range(self.n_nodes)]))

    def function_generator(self, predecessors, target):
        # for i in predecessors:
        #     print(self.dag())
        # pdb.set_trace()

        def my_func(predecessors):
            return 0
        return my_func

    def draw(self):
        """
        Draw the DAG with matplotlib
        """
        random.seed(0)
        # positions for all nodes, chosen in a deterministic way
        pos = nx.spring_layout(self.dag, pos=dict(zip(range(self.n_nodes), [(random.random(),
                                                                             random.random())
                                                                            for i in range(self.n_nodes)])))
        nx.draw_networkx_nodes(self.dag, pos)
        nx.draw_networkx_edges(self.dag, pos)
        plt.show()


if __name__ == "__main__":
    G = DirectedAcyclicGraph(3)
    G.draw()
