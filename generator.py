import numpy as np
import networkx as nx


class Generator:
    def __init__(self, n_latent, n_genes):
        self.n_latent = n_latent
        self.n_genes = n_genes

        self.latent_dag = 0
