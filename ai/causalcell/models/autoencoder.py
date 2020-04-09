from torch.nn.utils import spectral_norm
import ai.causalcell.utils.register as register
import torch
import torch.nn as nn


class Dummy(torch.nn.Module):
    """
    Helper class that allows you to do nothing. Replaces `lambda x: x`.
    """

    def forward(self, x):
        return x


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


@register.setmodelname('basic_AE')
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
        self.decoder = LinearLayers(layers=layers[::-1], dropout=dropout,
                                    norm=norm, activate_final=False)
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def embed(self, x):
        return self.encoder(x)

    def forward(self, x, fingerprint=0, compound=0, line=0):
        z = self.encoder(x)
        X_prime = self.decoder(z)
        return {'z': z, 'x_prime': X_prime, 'x': x}

    def loss(self, outputs):
        recon_loss = self.criterion(outputs['x_prime'], outputs['x'])
        return {'recon_loss': recon_loss}


@register.setmodelname('basic_VAE')
class VariationalAutoEncoder(nn.Module):
    """
    A basic feed forward classifier architecture with configurable dropout and
    layer-wise normalization with ~*~ variational inference ~*~.
    """
    def __init__(self, layers, beta=1, dropout=0, norm='none'):
        """
        An MLP variational antoencoder.
        """
        super(VariationalAutoEncoder, self).__init__()

        self.encoder = LinearLayers(layers=layers[:-1], dropout=dropout, norm=norm)
        self.mu = LinearLayers(layers=[layers[-2], layers[-1]], activate_final=False)
        self.logvar = LinearLayers(layers=[layers[-2], layers[-1]], activate_final=False)
        self.decoder = LinearLayers(layers=layers[::-1], dropout=dropout,
                                    norm=norm, activate_final=False)
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.beta = beta

    def embed(self, X):
        h = self.encoder(X)
        mu = self.mu(h)
        logvar = self.logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        z = mu + (std * eps)

        return z

    def decode(self, z):
        x_prime = self.decoder(z)

        return x_prime

    def forward(self, x, fingerprint=0, compound=0, line=0):
        mu, logvar = self.embed(x)
        z = self.reparameterize(mu, logvar)
        x_prime = self.decoder(z)
        return {'z': z, 'x_prime': x_prime, 'x': x, 'mu': mu, 'logvar': logvar}

    def loss(self, outputs):
        recon_loss = self.criterion(outputs['x_prime'], outputs['x'])
        kl_div = -0.5 * torch.sum(1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp())

        # Apply beta scaling factor
        kl_div *= self.beta

        return {'recon_loss': recon_loss, 'kl_div': kl_div}
