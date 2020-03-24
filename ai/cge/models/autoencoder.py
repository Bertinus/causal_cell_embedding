from ai.cge.models.utils import generate_feedforward_layers, Dummy
from copy import copy
from torch.nn.utils import spectral_norm
import ai.semrep.utils.register as register
import importlib
import torch
import torch.nn as nn
import numpy as np


class LinearBatch(nn.Module):
    """A linear layer with batch normalization."""
    def __init__(self, input_dim, output_dim, bias=True):
        super(LinearBatch, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        out = self.linear(x)
        return self.batch_norm(out)


class LinearSpec(nn.Module):
    """A linear layer with spectral normalization."""
    def __init__(self, input_dim, output_dim, bias=True):
        super(LinearSpec, self).__init__()
        self.linear = spectral_norm(nn.Linear(input_dim, output_dim, bias=bias))

    def forward(self, x):
        return self.linear(x)


class LinearLayers(nn.Module):
    """
    A basic feed forward network with configurable dropout and layer-wise
    normalization.
    """
    def __init__(self, layers, dropout=0, norm='none', modulelist=False,
                 bias=True, activate_final=True):
        super(LinearLayers, self).__init__()

        norm_dict = {
            'batch': LinearBatch, 'spectral': LinearSpec, 'none': nn.Linear}

        assert 0 <= dropout < 1
        assert norm in norm_dict.keys()

        # Set up the linear layers.
        self.linear = norm_dict[norm]
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.activations = []

        arch = []
        n_layers = len(layers)
        for i in range(n_layers-1):
            arch.append(self.linear(layers[i], layers[i+1], bias=bias))

            # Add activation to all layers, except the final one optionally.
            if activate_final or i+1 < n_layers-1:
                arch.append(self.activation)

                # Only dropout the activated layers.
                if dropout > 0:
                    arch.append(self.dropout)

        # Allows for the linear layer to be a passthrough in the case that
        # the number of layers requested is empty or a single value.
        if len(arch) > 0:
            if modulelist:
                self.model = nn.ModuleList(arch)
            else:
                self.model = nn.Sequential(*arch)
        else:
            self.model = Dummy()

    def embed(self, X):
        return self.model(X)

    def forward(self, X):
        return self.embed(X)


class LinearActivationSaver(LinearLayers):
    """
    A linear classifier that also saves all activations on the forward pass.
    Useful for actdiff regularization or enforcing representation sparsity.
    """
    def __init__(self, layers, dropout=0, norm='none', activate_final=True,
                 store_pre=False):
        super(LinearActivationSaver, self).__init__(
            layers=layers, dropout=dropout, norm=norm,
            activate_final=activate_final, modulelist=True)

        # Store the values either before or after ReLU.
        if store_pre:
            self.instances = (LinearBatch, LinearSpec, nn.Linear)
        else:
            self.instances = (nn.ReLU)

    def embed(self, X):
        self.activations = []

        # Loop through layers, saving all pre/post activations.
        for layer in self.model:
            X = layer(X)
            if isinstance(layer, self.instances):
                self.activations.append(X)

        return X


class AutoEncoder(nn.Module):
    """
    A basic feed forward classifier architecture with configurable dropout and
    layer-wise normalization.
    """
    def __init__(self, layers, dropout=0, norm='none'):
        """
        An MLP vanilla antoencoder.
        """
        super(AutoEncoder, self).__init__()

        # assert num_classes >= 1

        self.encoder = LinearLayers(layers=layers, dropout=dropout, norm=norm)
        self.decoder = LinearLayers(layers=layers.reverse(), dropout=dropout,
                                    norm=norm)
        self.criterion = torch.nn.MSELoss()

    def embed(self, X):
        return self.encoder(X)

    def forward(self, X):
        z = self.encoder(X)
        X_prime = self.decoder(z)
        return {'z': z, 'X_prime': X_prime, 'X': X}

    def loss(self, y, outputs):
        recon_loss = self.criterion(outputs['X_prime'], outputs['X'])
        return {'recon_loss': recon_loss}


class VariationalAutoEncoder(nn.Module):
    """
    A basic feed forward classifier architecture with configurable dropout and
    layer-wise normalization with ~*~ variational inference ~*~.
    """
    def __init__(self, layers, z_size=20, dropout=0, norm='none'):
        """
        An MLP variational antoencoder.
        """
        super(VariationalAutoEncoder, self).__init__()

        # assert num_classes >= 1

        self.encoder = LinearLayers(layers=layers, dropout=dropout, norm=norm)
        self.mu = LinearLayers(layers=[layers[-1], z_size])
        self.logvar = LinearLayers(layers=[layers[-1], z_size])
        self.decoder = LinearLayers(layers=layers.reverse(), dropout=dropout,
                                    norm=norm)
        self.criterion = torch.nn.MSELoss()

    def embed(self, X):
        h = self.encoder(X)
        mu = self.mu(h)
        logvar = self.logvar(h)

        return (mu, logvar)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        z = mu + (std * eps)

        return z

    def decode(self, z):
        X_prime = self.decoder(z)

        return X_prime

    def forward(self, X):
        mu, logvar = self.embed(X)
        z = self.reparameterize(mu, logvar)
        X_prime = self.decoder(z)
        return {'z': z, 'X_prime': X_prime, 'X': X}

    def loss(self, y, outputs):
        recon_loss = self.criterion(outputs['X_prime'], outputs['X'])
        return {'recon_loss': recon_loss}


class VariationalAutoEncoder(AutoEncoder):
    """"""""
    def __init__(self, layers, dropout=0, norm='none'):
        super(VariationalAutoEncoder, self).__init__(
            layers, dropout=dropout, norm=norm)
