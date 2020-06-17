from ai.causalcell.models.utils import *
import ai.causalcell.utils.register as register
import torch
import torch.nn as nn
import copy


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
        z = self.embed(x)
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


@register.setmodelname('low_var_VAE')
class LowVarianceVariationalAutoEncoder(VariationalAutoEncoder):
    def __init__(self, enc_layers, dec_layers, prior_variance, beta=1, dropout=0, norm='none'):
        # Define layers for the encoder and decoder, which have to take environment in input as well
        self.prior_variance = torch.Tensor([prior_variance])
        super(LowVarianceVariationalAutoEncoder, self).__init__(enc_layers, dec_layers, beta=beta,
                                                                dropout=dropout, norm=norm)

    def loss(self, outputs):
        recon_loss = self.criterion(outputs['x_prime'], outputs['x'])
        # The KL div is computed wrt to N(0, prior_variance)
        kl_div = - 0.5 * torch.sum(1 - self.prior_variance.log() + outputs['logvar']
                                   - 1 / self.prior_variance * (outputs['mu']).pow(2)
                                   - outputs['logvar'].exp() / self.prior_variance)

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
        self.enc_layers = copy.deepcopy(enc_layers)
        self.dec_layers = copy.deepcopy(dec_layers)
        self.enc_layers[0] += condition_dim
        super(EnvironmentInputVariationalAutoEncoder, self).__init__(self.enc_layers, self.dec_layers, beta=beta,
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
        self.enc_layers = copy.deepcopy(enc_layers)
        self.dec_layers = copy.deepcopy(dec_layers)
        # Define layers for the encoder and decoder, which have to take environment in input as well
        self.enc_layers[0] += condition_dim
        self.dec_layers[0] += condition_dim

        super(ConditionalVariationalAutoEncoder, self).__init__(self.enc_layers, self.dec_layers,
                                                                beta=beta, dropout=dropout, norm=norm)

    def forward(self, x, fingerprint, compound=0, line=0):
        mu, logvar = self.embed(torch.cat((x, fingerprint), dim=1))
        z = self.reparameterize(mu, logvar)
        x_prime = self.decoder(torch.cat((z, fingerprint), dim=1))
        return {'z': z, 'x_prime': x_prime, 'x': x, 'mu': mu, 'logvar': logvar}


@register.setmodelname('mean_activation_model')
class MeanActivationModel(nn.Module):
    """
    A simple model that predicts the average activation (on the data seen so far) for each gene
    """

    def __init__(self):
        super(MeanActivationModel, self).__init__()
        self.n_examples_seen_so_far = 0
        self.means = None
        # self.encoder = LinearLayers(layers=[1, 1])  # Just define this to avoid issues in the main pipeline
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def forward(self, x, fingerprint, compound=0, line=0):
        batch_means = torch.mean(x, dim=0)
        if self.means is None:
            self.means = batch_means
            self.n_examples_seen_so_far = x.shape[0]
        else:
            self.means = (self.n_examples_seen_so_far * self.means + x.shape[0] * batch_means) / \
                         (self.n_examples_seen_so_far + x.shape[0])
            self.n_examples_seen_so_far += x.shape[0]
        x_prime = self.means[None, :].repeat(x.shape[0], 1)
        return {'x': x, 'x_prime': x_prime}

    def loss(self, outputs):
        recon_loss = self.criterion(outputs['x_prime'], outputs['x'])
        return {'recon_loss': recon_loss}

# TODO: add an environment specific mean activation model
