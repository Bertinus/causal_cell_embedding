from ai.causalcell.models.utils import *
import ai.causalcell.utils.register as register
import torch
from ai.causalcell.models.env_prior_models import EnvironmentPriorVariationalAutoEncoder


@register.setmodelname('adv_VAE')
class AdversarialVariationalAutoEncoder(EnvironmentPriorVariationalAutoEncoder):
    """
    VAE with an adversarial setting
    """

    def __init__(self, enc_layers, dec_layers, aux_layers, adv_layers, beta=1, dropout=0, norm='none', softmax=True,
                 temperature=1):
        super(AdversarialVariationalAutoEncoder, self).__init__(enc_layers, dec_layers, aux_layers, beta=beta,
                                                                dropout=dropout, norm=norm, softmax=softmax,
                                                                temperature=temperature)

        self.adversarial_net = LinearLayers(layers=adv_layers, dropout=dropout, norm=norm, activate_final=False)

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
        # Adversarial
        predicted_env = self.adversarial_net(z)

        return {'z': z, 'x_prime': x_prime, 'x': x, 'mu': mu, 'logvar': logvar,
                'env_mu': env_mu, 'env_logvar': env_logvar,
                'fingerprint': fingerprint, 'predicted_env': predicted_env}

    def loss(self, outputs):
        recon_loss = self.criterion(outputs['x_prime'], outputs['x'])

        # The KL div is computed wrt the environment specific prior
        kl_div = - 0.5 * torch.sum(1 - outputs['env_logvar'] + outputs['logvar']
                                   - 1 / (outputs['env_logvar'].exp()) * (outputs['mu'] - outputs['env_mu']).pow(2)
                                   - outputs['logvar'].exp() / outputs['env_logvar'].exp())

        # Apply beta scaling factor
        kl_div *= self.beta

        # Adversarial loss
        adv_loss = self.criterion(outputs['predicted_env'], outputs['fingerprint'])

        return {'recon_loss': recon_loss, 'kl_div': kl_div}