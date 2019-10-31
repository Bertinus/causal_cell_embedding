"""
This file contains the functions that are possible to use in structural equations
as well as the StructuralEquation class
"""
import numpy as np


class StructuralFunction:
    def __init__(self, **kwargs):
        pass

    def compute(self, inputs, edge_weights=None):
        """
        :param inputs: list of neighbours values
        :param edge_weights: list of edge weights
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

    def compute(self, inputs, edge_weights=None):
        if not inputs:
            return 0
        return max(inputs) + self.offset


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


########################################################################################################################
# Other types of structural functions
########################################################################################################################

class Linear(StructuralFunction):
    def __init__(self, offset=0):
        super().__init__()
        self.offset = offset
        return

    def __str__(self):
        return "linear"

    def __repr__(self):
        return "linear"

    def compute(self, inputs, edge_weights=None):
        if not edge_weights:
            return 0
        inputs = np.array(inputs)
        edge_weights = np.array(edge_weights)
        return int((inputs * edge_weights).sum()) + self.offset


class NoisyLinear(StructuralFunction):
    def __init__(self, sampler=lambda: np.random.normal(loc=0.0, scale=1.0, size=None), offset=0):
        super().__init__()
        self.sampler = sampler
        self.offset = offset
        return

    def __str__(self):
        return "noisy linear"

    def __repr__(self):
        return "noisy linear"

    def compute(self, inputs, edge_weights=None):
        if not edge_weights:
            return self.sampler()
        inputs = np.array(inputs)
        edge_weights = np.array(edge_weights)
        return int((inputs * edge_weights).sum()) + self.offset + self.sampler()


if __name__ == "__main__":
    noisylin = NoisyLinear()
    print(type(noisylin.compute(0)))
    print(noisylin.compute(0))
    print(noisylin.compute(0))
