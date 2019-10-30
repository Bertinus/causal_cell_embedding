"""
This file contains the functions that are possible to use in structural equations
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
