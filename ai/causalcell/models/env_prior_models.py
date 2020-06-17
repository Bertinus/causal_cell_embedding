from ai.causalcell.models.utils import *
import ai.causalcell.utils.register as register
import torch
from ai.causalcell.models.autoencoder import VariationalAutoEncoder
import torch.nn as nn
import copy


@register.setmodelname('env_mu_prior_VAE')
class EnvironmentMuPriorVariationalAutoEncoder(VariationalAutoEncoder):
    """
    VAE that adapts to each environment by modifying the mean of the prior in latent space based on environment
    """

    def __init__(self, enc_layers, dec_layers, aux_layers, beta=1, dropout=0, norm='none', softmax=True,
                 temperature=1):
        self.enc_layers = copy.deepcopy(enc_layers)
        self.dec_layers = copy.deepcopy(dec_layers)
        self.enc_layers[0] += aux_layers[0]  # The encoder has to take the fingerprint as input
        super(EnvironmentMuPriorVariationalAutoEncoder, self).__init__(self.enc_layers, self.dec_layers,
                                                                       beta=beta, dropout=dropout, norm=norm)
        if softmax:
            self.softmax = nn.Softmax(dim=1)
            self.temperature = temperature
        else:
            if temperature != 1:
                print("If Softmax is False, Temperature is set to 1")
            self.temperature = 1
            self.softmax = Dummy()

        self.env_prior_mu = LinearLayers(layers=aux_layers, dropout=dropout, norm=norm, activate_final=False)

    def forward(self, x, fingerprint, compound=0, line=0):
        mu, logvar = self.embed(torch.cat((x, fingerprint), dim=1))
        env_mu = self.env_prior_mu(fingerprint)
        alpha = self.softmax(1 / self.temperature * torch.abs(env_mu))
        env_mu = alpha * env_mu  # Make mu sparse
        z = self.reparameterize(mu, logvar)
        x_prime = self.decoder(z)
        return {'z': z, 'x_prime': x_prime, 'x': x, 'mu': mu, 'logvar': logvar,
                'env_mu': env_mu}

    def loss(self, outputs):
        recon_loss = self.criterion(outputs['x_prime'], outputs['x'])

        # The KL div is computed wrt the environment specific prior
        kl_div = - 0.5 * torch.sum(1 + outputs['logvar'] - (outputs['mu'] - outputs['env_mu']).pow(2)
                                   - outputs['logvar'].exp())

        # Apply beta scaling factor
        kl_div *= self.beta

        return {'recon_loss': recon_loss, 'kl_div': kl_div}


@register.setmodelname('env_prior_VAE')
class EnvironmentPriorVariationalAutoEncoder(EnvironmentMuPriorVariationalAutoEncoder):
    """
    VAE that adapts to each environment by modifying the mean and variance of the prior
    in latent space based on environment
    """

    def __init__(self, enc_layers, dec_layers, aux_layers, beta=1, dropout=0, norm='none', softmax=True,
                 temperature=1):
        super(EnvironmentPriorVariationalAutoEncoder, self).__init__(enc_layers, dec_layers, aux_layers, beta=beta,
                                                                     dropout=dropout, norm=norm, softmax=softmax,
                                                                     temperature=temperature)
        self.env_prior_logvar = LinearLayers(layers=aux_layers, dropout=dropout,
                                             norm=norm, activate_final=False)

    def forward(self, x, fingerprint, compound=0, line=0):
        mu, logvar = self.embed(torch.cat((x, fingerprint), dim=1))
        env_mu = self.env_prior_mu(fingerprint)
        env_logvar = self.env_prior_logvar(fingerprint)
        # Make mu sparse
        alpha_mu = self.softmax(1 / self.temperature * torch.abs(env_mu))
        env_mu = alpha_mu * env_mu
        # Make logvar sparse
        alpha_var = self.softmax(1 / self.temperature * torch.abs(env_logvar))
        env_logvar = alpha_var * env_logvar
        z = self.reparameterize(mu, logvar)
        x_prime = self.decoder(z)
        return {'z': z, 'x_prime': x_prime, 'x': x, 'mu': mu, 'logvar': logvar,
                'env_mu': env_mu, 'env_logvar': env_logvar}

    def loss(self, outputs):
        recon_loss = self.criterion(outputs['x_prime'], outputs['x'])

        # The KL div is computed wrt the environment specific prior
        kl_div = - 0.5 * torch.sum(1 - outputs['env_logvar'] + outputs['logvar']
                                   - 1 / (outputs['env_logvar'].exp()) * (outputs['mu'] - outputs['env_mu']).pow(2)
                                   - outputs['logvar'].exp() / outputs['env_logvar'].exp())

        # Apply beta scaling factor
        kl_div *= self.beta

        return {'recon_loss': recon_loss, 'kl_div': kl_div}
