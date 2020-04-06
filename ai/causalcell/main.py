import os
import click
import ai.causalcell.training as training
import ai.causalcell.utils.configuration as configuration

# Enable click
os.environ['LC_ALL'] = 'C.UTF-8'
os.environ['LANG'] = 'C.UTF-8'


@click.group()
def run():
    pass


@run.command()
@click.option('--config', '-cgf',
              type=click.Path(exists=True, resolve_path=True),
              help='Configuration file.')
def train(config):
    cfg = configuration.load_config(config)
    training.train(cfg)


def main():
    run()


if __name__ == '__main__':
    train()
