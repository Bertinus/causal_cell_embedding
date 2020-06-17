from ai.causalcell.models.utils import *
import ai.causalcell.utils.register as register
import torch
from ai.causalcell.models.autoencoder import VariationalAutoEncoder
import torch.nn as nn
import copy


@register.setmodelname('env_trans_VAE')
class TranslationVariationalAutoEncoder(VariationalAutoEncoder):
    """
    VAE that adapts to each environment by translating one of the latent variables.
    The prior in latent space depends on the translation (after the translation is applied)
    """

    def __init__(self, enc_layers, dec_layers, aux_layers, beta=1, dropout=0, norm='none', softmax=True,
                 temperature=1):
        """
        :param softmax: if True, a softmax is used to normalize env_mu so that the absolute values of env_mu sum to 1
        :param temperature: temperature of the softmax
        """
        self.enc_layers = copy.deepcopy(enc_layers)
        self.dec_layers = copy.deepcopy(dec_layers)
        self.enc_layers[0] += aux_layers[0]  # The encoder has to take the fingerprint as input
        super(TranslationVariationalAutoEncoder, self).__init__(self.enc_layers, self.dec_layers,
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
        x_prime = self.decoder(z + env_mu)
        return {'z': z, 'x_prime': x_prime, 'x': x, 'mu': mu, 'logvar': logvar,
                'env_mu': env_mu}


@register.setmodelname('no_env_input_trans_VAE')
class NoEnvInputTranslationVariationalAutoEncoder(VariationalAutoEncoder):
    """
    VAE that adapts to each environment by translating one of the latent variables.
    The prior in latent space depends on the translation (after the translation is applied)
    """

    def __init__(self, enc_layers, dec_layers, aux_layers, beta=1, dropout=0, norm='none', softmax=True,
                 temperature=1):
        """
        :param softmax: if True, a softmax is used to normalize env_mu so that the absolute values of env_mu sum to 1
        :param temperature: temperature of the softmax
        """
        super(NoEnvInputTranslationVariationalAutoEncoder, self).__init__(enc_layers, dec_layers,
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
        mu, logvar = self.embed(x)
        env_mu = self.env_prior_mu(fingerprint)
        alpha = self.softmax(1 / self.temperature * torch.abs(env_mu))
        env_mu = alpha * env_mu  # Make mu sparse
        z = self.reparameterize(mu, logvar)
        x_prime = self.decoder(z + env_mu)
        return {'z': z, 'x_prime': x_prime, 'x': x, 'mu': mu, 'logvar': logvar,
                'env_mu': env_mu}
