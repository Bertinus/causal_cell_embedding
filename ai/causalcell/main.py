import os
import click
import ai.causalcell.training as training
import ai.causalcell.utils.configuration as configuration

# Enable click
os.environ['LC_ALL'] = 'C.UTF-8'
os.environ['LANG'] = 'C.UTF-8'


def load_ith_value_of_generic_dict(d_config, d_generic, i):
    if isinstance(d_generic, dict):  # Apply recursively
        for key in d_generic.keys():
            d_config[key] = load_ith_value_of_generic_dict(d_config[key], d_generic[key], i)
        return d_config
    else:  # d_generic should be a list
        d_config = d_generic[i]
        return d_config


@click.group()
def run():
    pass


@run.command()
@click.option('--config', '-cgf',
              type=click.Path(exists=True, resolve_path=True),
              help='Configuration file.')
def train(config):
    cfg = configuration.load_config(config)
    if cfg['generic']:
        for i in range(len(cfg['generic']['exp_id'])):
            cfg = load_ith_value_of_generic_dict(cfg, cfg['generic'], i)
            training.train(cfg)
    else:
        training.train(cfg)


if __name__ == '__main__':
    train()
