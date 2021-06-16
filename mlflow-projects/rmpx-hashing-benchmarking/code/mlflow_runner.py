import mlflow
import click

from train import run

mlflow.set_tracking_uri("http://localhost/mlflow")


@click.command()
@click.option("--rmpx-size", type=click.Choice(['3', '6', '11', '20', '37']),
              required=True)
@click.option("--trials", type=click.INT, default=100)
@click.option("--hash",
              type=click.Choice(['SHA256'], case_sensitive=False),
              required=True)
@click.option("--agent",
              type=click.Choice(['ACS', 'ACS2', 'ACS2GA', 'YACS'],
                                case_sensitive=False),
              required=True)
def execute(rmpx_size, trials, hash, agent):
    rmpx_size = int(rmpx_size)

    mlflow.log_param("rmpx_size", rmpx_size)
    mlflow.log_param("trials", trials)
    mlflow.log_param("hash", hash)
    mlflow.log_param("agent", agent)

    run(rmpx_size, trials, agent, hash, 16)


if __name__ == '__main__':
    execute()
