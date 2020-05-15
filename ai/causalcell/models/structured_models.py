from ai.causalcell.models.utils import *
import ai.causalcell.utils.register as register
import torch
from ai.causalcell.models.env_prior_models import EnvironmentMuPriorVariationalAutoEncoder

# TODO: Modified the generate_causes methods to allow to generate several causal variables at the same time


@register.setmodelname('structured_prior_VAE')
class StructuredPriorVariationalAutoEncoder(EnvironmentMuPriorVariationalAutoEncoder):
    """
    VAE that has some structure in latent space and some environment specific mean for the prior
    """

    def __init__(self, enc_layers, dec_layers, aux_layers, beta=1, dropout=0, norm='none', softmax=False,
                 temperature=1):
        super(StructuredPriorVariationalAutoEncoder, self).__init__(enc_layers, dec_layers, aux_layers, beta=beta,
                                                                    dropout=dropout, norm=norm, softmax=softmax,
                                                                    temperature=temperature)
        self.latent_dim = enc_layers[-1]
        self.latent_processes = []
        for i in range(self.latent_dim):
            self.latent_processes.append(
                LinearLayers(layers=[i + 1, 1], dropout=dropout, norm=norm, activate_final=False))

        self.latent_processes = nn.ModuleList(self.latent_processes)

    def generate_causes(self, z):
        c = self.latent_processes[0](z[:, 0:1])
        for i in range(1, self.latent_dim):
            ith_input = torch.cat((c, z[:, i:i + 1]), dim=1)
            c_i = self.latent_processes[i](ith_input)
            c = torch.cat((c, c_i), dim=1)
        return c

    def forward(self, x, fingerprint, compound=0, line=0):
        mu, logvar = self.embed(torch.cat((x, fingerprint), dim=1))
        env_mu = self.env_prior_mu(fingerprint)
        # Make mu sparse
        alpha_mu = self.softmax(1 / self.temperature * torch.abs(env_mu))
        env_mu = alpha_mu * env_mu
        z = self.reparameterize(mu, logvar)
        c = self.generate_causes(z)
        x_prime = self.decoder(c)
        return {'z': z, 'c': c, 'x_prime': x_prime, 'x': x, 'mu': mu, 'logvar': logvar,
                'env_mu': env_mu}


@register.setmodelname('structured_trans_VAE')
class StructuredTranslationVariationalAutoEncoder(EnvironmentMuPriorVariationalAutoEncoder):
    """
    VAE that has some structure in latent space and some environment specific translation
    """

    def __init__(self, enc_layers, dec_layers, aux_layers, beta=1, dropout=0, norm='none', softmax=False,
                 temperature=1):
        super(StructuredTranslationVariationalAutoEncoder, self).__init__(enc_layers, dec_layers, aux_layers, beta=beta,
                                                                          dropout=dropout, norm=norm, softmax=softmax,
                                                                          temperature=temperature)
        self.latent_dim = enc_layers[-1]
        self.latent_processes = []
        for i in range(self.latent_dim):
            self.latent_processes.append(
                LinearLayers(layers=[i + 1, 1], dropout=dropout, norm=norm, activate_final=False))

        self.latent_processes = nn.ModuleList(self.latent_processes)

    def generate_causes(self, z):
        c = self.latent_processes[0](z[:, 0:1])
        for i in range(1, self.latent_dim):
            ith_input = torch.cat((c, z[:, i:i + 1]), dim=1)
            c_i = self.latent_processes[i](ith_input)
            c = torch.cat((c, c_i), dim=1)
        return c

    def forward(self, x, fingerprint, compound=0, line=0):
        mu, logvar = self.embed(torch.cat((x, fingerprint), dim=1))
        env_mu = self.env_prior_mu(fingerprint)
        alpha = self.softmax(1 / self.temperature * torch.abs(env_mu))
        env_mu = alpha * env_mu  # Make mu sparse
        z = self.reparameterize(mu, logvar)
        c = self.generate_causes(z + env_mu)
        x_prime = self.decoder(c)
        return {'z': z, 'c': c, 'x_prime': x_prime, 'x': x, 'mu': mu, 'logvar': logvar,
                'env_mu': env_mu}
