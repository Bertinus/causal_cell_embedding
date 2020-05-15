from ai.causalcell.models.utils import *
import ai.causalcell.utils.register as register
import torch
import torch.nn as nn


@register.setmodelname('basic_AE')
class AutoEncoder(nn.Module):
    """
    A basic feed forward classifier architecture with configurable dropout and
    layer-wise normalization.
    """

    def __init__(self, enc_layers, dec_layers, dropout=0, norm='none'):
        """
        An MLP vanilla antoencoder.
        """
        super(AutoEncoder, self).__init__()

        # assert num_classes >= 1
        self.encoder = LinearLayers(layers=enc_layers, dropout=dropout, norm=norm, activate_final=False)
        self.decoder = LinearLayers(layers=dec_layers, dropout=dropout,
                                    norm=norm, activate_final=False)
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def embed(self, x):
        return self.encoder(x)

    def forward(self, x, fingerprint=0, compound=0, line=0):
        z = self.encoder(x)
        x_prime = self.decoder(z)
        return {'z': z, 'x_prime': x_prime, 'x': x}

    def loss(self, outputs):
        recon_loss = self.criterion(outputs['x_prime'], outputs['x'])
        return {'recon_loss': recon_loss}


@register.setmodelname('basic_VAE')
class VariationalAutoEncoder(nn.Module):
    """
    A basic feed forward classifier architecture with configurable dropout and
    layer-wise normalization with ~*~ variational inference ~*~.
    """

    def __init__(self, enc_layers, dec_layers, beta=1, dropout=0, norm='none'):
        """
        An MLP variational antoencoder.
        """
        super(VariationalAutoEncoder, self).__init__()

        self.encoder = LinearLayers(layers=enc_layers[:-1], dropout=dropout, norm=norm)
        self.mu = LinearLayers(layers=[enc_layers[-2], enc_layers[-1]], activate_final=False)
        self.logvar = LinearLayers(layers=[enc_layers[-2], enc_layers[-1]], activate_final=False)
        self.decoder = LinearLayers(layers=dec_layers, dropout=dropout,
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

    def forward(self, x, fingerprint, compound=0, line=0):
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


@register.setmodelname('env_input_VAE')
class EnvironmentInputVariationalAutoEncoder(VariationalAutoEncoder):
    """
    VAE that takes environment as input in the encoder only
    """

    def __init__(self, enc_layers, dec_layers, condition_dim, beta=1, dropout=0, norm='none'):
        # Define layers for the encoder and decoder, which have to take environment in input as well
        enc_layers[0] += condition_dim
        super(EnvironmentInputVariationalAutoEncoder, self).__init__(enc_layers, dec_layers, beta=beta,
                                                                     dropout=dropout, norm=norm)

    def forward(self, x, fingerprint, compound=0, line=0):
        mu, logvar = self.embed(torch.cat((x, fingerprint), dim=1))
        z = self.reparameterize(mu, logvar)
        x_prime = self.decoder(z)
        return {'z': z, 'x_prime': x_prime, 'x': x, 'mu': mu, 'logvar': logvar}


@register.setmodelname('conditional_VAE')
class ConditionalVariationalAutoEncoder(VariationalAutoEncoder):
    """
    A MLP conditional VAE
    """
    def __init__(self, enc_layers, dec_layers, condition_dim, beta=1, dropout=0, norm='none'):

        # Define layers for the encoder and decoder, which have to take environment in input as well
        enc_layers[0] += condition_dim
        dec_layers[0] += condition_dim

        super(ConditionalVariationalAutoEncoder, self).__init__(enc_layers, dec_layers,
                                                                beta=beta, dropout=dropout, norm=norm)

    def forward(self, x, fingerprint, compound=0, line=0):
        mu, logvar = self.embed(torch.cat((x, fingerprint), dim=1))
        z = self.reparameterize(mu, logvar)
        x_prime = self.decoder(torch.cat((z, fingerprint), dim=1))
        return {'z': z, 'x_prime': x_prime, 'x': x, 'mu': mu, 'logvar': logvar}
